# Copyright (c) Alibaba, Inc. and its affiliates.

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from peft import PeftModel
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers import Trainer as HfTrainer
from transformers import trainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available

from swift.torchacc_utils import ta_eval_dataloader, ta_test_dataloader, ta_train_dataloader, ta_trim_graph
from swift.utils import use_torchacc
from .callback import DefaultFlowCallbackNew, PrinterCallbackNew, ProgressCallbackNew
from .mixin import PushToMsHubMixin, SwiftMixin

try:
    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
except ImportError:
    from transformers.deepspeed import is_deepspeed_zero3_enabled


class Trainer(PushToMsHubMixin, SwiftMixin, HfTrainer):
    pass


class Seq2SeqTrainer(PushToMsHubMixin, SwiftMixin, HfSeq2SeqTrainer):

    def __init__(self, *args, **kwargs):
        self.sequence_parallel_size = kwargs.pop('sequence_parallel_size', 1)
        super().__init__(*args, **kwargs)
        # performance
        if not hasattr(self, 'perf'):
            self.perf = {}
        self.perf.update({
            'gen_time': 0.,
            'gen_len': 0,
        })
        self._pos_acc = torch.tensor(0.).to(self.args.device)
        self._neg_acc = torch.tensor(0.).to(self.args.device)
        if self.sequence_parallel_size > 1:
            from swift.trainers.xtuner import init_sequence_parallel_xtuner
            init_sequence_parallel_xtuner(self.sequence_parallel_size)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)

        inputs.pop('loss_scale', None)
        has_labels = 'labels' in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        # Priority (handled in generate):
        # gen_kwargs > model.generation_config > default GenerationConfig()

        if len(gen_kwargs) == 0 and hasattr(self, '_gen_kwargs'):
            gen_kwargs = self._gen_kwargs.copy()
            if hasattr(self.model, 'generation_config'):
                gen_kwargs.update(self.model.generation_config.to_dict())

        if gen_kwargs.get('max_length') is None and gen_kwargs.get('max_new_tokens') is None:
            gen_kwargs['max_length'] = self.model.config.max_length
        gen_kwargs['num_beams'] = (
            gen_kwargs['num_beams'] if gen_kwargs.get('num_beams') is not None else self.model.config.num_beams)
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs['synced_gpus'] = (
            gen_kwargs['synced_gpus'] if gen_kwargs.get('synced_gpus') is not None else default_synced_gpus)

        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if ('labels' in inputs and 'decoder_input_ids' in inputs
                and inputs['labels'].shape == inputs['decoder_input_ids'].shape):
            inputs = {k: v for k, v in inputs.items() if k != 'decoder_input_ids'}

        gen_kwargs['pad_token_id'] = self.tokenizer.pad_token_id
        gen_kwargs['eos_token_id'] = self.tokenizer.eos_token_id
        # fix generate warning
        if 'max_length' in gen_kwargs and 'max_new_tokens' in gen_kwargs and gen_kwargs['max_new_tokens'] is not None:
            gen_kwargs.pop('max_length')
        gen_time = time.time()
        generate_inputs = inputs.copy()
        if has_labels:
            _labels = inputs['labels'][0]
            n_mask = 0
            for i in range(len(_labels)):
                if _labels[i] != -100:
                    n_mask = i
                    break

            for k in ['input_ids', 'attention_mask']:
                generate_inputs[k] = generate_inputs[k][:, :n_mask]
            generate_inputs['labels'] = generate_inputs['labels'][:, n_mask:]

        generated_tokens = self.model.generate(**generate_inputs, **gen_kwargs)
        gen_time = time.time() - gen_time

        if hasattr(self.model, 'encoder') and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = generate_inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = generate_inputs[self.model.main_input_name]

        generated_tokens = generated_tokens[:, generation_inputs.shape[1]:]
        gen_len = len(generated_tokens[0])
        self.perf['gen_time'] = self.perf['gen_time'] + gen_time
        self.perf['gen_len'] = self.perf['gen_len'] + gen_len

        # in case the batch is shorter than max length, the output should be padded
        if gen_kwargs.get('max_length') is not None and generated_tokens.shape[-1] < gen_kwargs['max_length']:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs['max_length'])
        elif gen_kwargs.get('max_new_tokens') is not None and generated_tokens.shape[-1] < (gen_kwargs['max_new_tokens']
                                                                                            + 1):
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs['max_new_tokens'] + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs['labels']).mean().detach()
                else:
                    loss = (outputs['loss'] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = generate_inputs['labels']
            if gen_kwargs.get('max_length') is not None and labels.shape[-1] < gen_kwargs['max_length']:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs['max_length'])
            elif gen_kwargs.get('max_new_tokens') is not None and labels.shape[-1] < (gen_kwargs['max_new_tokens'] + 1):
                labels = self._pad_tensors_to_max_len(labels, (gen_kwargs['max_new_tokens'] + 1))
        else:
            labels = None

        return loss, generated_tokens, labels

    @staticmethod
    def compute_scaled_loss(labels: torch.Tensor, lm_logits: torch.Tensor, loss_scale: torch.Tensor) -> torch.Tensor:
        device = lm_logits.device
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        shift_scale = loss_scale[..., 1:]
        # Save memory
        masks = shift_labels != -100
        shift_logits = shift_logits[masks]
        shift_labels = shift_labels[masks].to(device)
        shift_scale = shift_scale[masks].to(device)
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits, shift_labels)
        loss = shift_scale * loss
        return loss.mean()
    
    @staticmethod
    def compute_margin_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor, pos_labels: torch.Tensor, neg_labels: torch.Tensor, tokenizer, margin: float = 0.2) -> torch.Tensor:
        # logits: bsz, seq_len, vocab_size
        # labels: bsz, seq_len
        device = pos_logits.device
        label_ids = [x[-1] for x in tokenizer(['0', '1']).input_ids]
        level0, level1 = label_ids
        # Shift labels to align with logits
        shift_pos_labels = pos_labels[..., 1:]
        shift_neg_labels = neg_labels[..., 1:]
        
        # Get the scores for the token_id at the label positions
        pos_scores_level1 = pos_logits[..., :-1, level1]
        neg_scores_level1 = neg_logits[..., :-1, level1]
        pos_scores_level0 = pos_logits[..., :-1, level0]
        neg_scores_level0 = neg_logits[..., :-1, level0]
        
        # Mask to ignore padding tokens
        pos_masks = shift_pos_labels != -100
        neg_masks = shift_neg_labels != -100
        pos_scores_level1 = pos_scores_level1[pos_masks]
        neg_scores_level1 = neg_scores_level1[neg_masks]
        pos_scores_level0 = pos_scores_level0[pos_masks]
        neg_scores_level0 = neg_scores_level0[neg_masks]
        
        # Compute margin loss for level1
        margin_loss_level1 = torch.clamp(margin - (pos_scores_level1 - neg_scores_level1), min=0.0)
        
        # Compute margin loss for level0
        margin_loss_level0 = torch.clamp(margin - (neg_scores_level0 - pos_scores_level0), min=0.0)
        
        # Combine the two margin losses
        total_margin_loss = margin_loss_level1.mean() + margin_loss_level0.mean()
        return total_margin_loss

    def compute_loss(self, model, inputs, return_outputs=None):
        # print(inputs.keys()) 
        # dict_keys(['pos_attention_mask', 'pos_labels', 'neg_attention_mask', 
        # 'neg_labels', 'pos_input_ids', 'neg_input_ids', 
        # 'pos_pixel_values', 'neg_pixel_values',
        #  'pos_image_flags', 'neg_image_flags'])
        pos_inputs = {
            'attention_mask': inputs['pos_attention_mask'],
            'labels': inputs['pos_labels'],
            'input_ids': inputs['pos_input_ids'],
            'pixel_values': inputs['pos_pixel_values'],
            'image_flags': inputs['pos_image_flags']
        }
        neg_inputs = {
            'attention_mask': inputs['neg_attention_mask'],
            'labels': inputs['neg_labels'],
            'input_ids': inputs['neg_input_ids'],
            'pixel_values': inputs['neg_pixel_values'],
            'image_flags': inputs['neg_image_flags']
        }
        if not hasattr(self, '_custom_metrics'):
            self._custom_metrics = {}

        pos_labels, neg_labels = None, None
        pos_loss_scale, neg_loss_scale = None, None
        if 'pos_loss_scale' in inputs:
            pos_labels = inputs.pop('pos_labels')
            pos_loss_scale = inputs.pop('pos_loss_scale')
        if 'neg_loss_scale' in inputs:
            neg_labels = inputs.pop('neg_labels')
            neg_loss_scale = inputs.pop('neg_loss_scale')

        if self.label_smoother is not None and 'pos_labels' in inputs:
            pos_labels = inputs.pop('pos_labels')
        if self.label_smoother is not None and 'neg_labels' in inputs:
            neg_labels = inputs.pop('neg_labels')

        pos_outputs = model(**pos_inputs)
        if pos_loss_scale is not None:
            # print("pos_labels",pos_labels)
            pos_outputs['loss'] = self.compute_scaled_loss(pos_labels, pos_outputs.logits, pos_loss_scale)
        neg_outputs = model(**neg_inputs)
        if neg_loss_scale is not None:
            neg_outputs['loss'] = self.compute_scaled_loss(neg_labels, neg_outputs.logits, neg_loss_scale)


        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past_pos = pos_outputs[self.args.past_index]
            self._past_neg = neg_outputs[self.args.past_index] # TODO: tbh: i don't know _past is what?

        if pos_labels is not None and pos_loss_scale is None:
            unwrapped_model = unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                pos_loss = self.label_smoother(pos_outputs, pos_labels, shift_labels=True)
            else:
                pos_loss = self.label_smoother(pos_outputs, pos_labels)
        else:
            # print("pos_labels None?")
            pos_loss = pos_outputs['loss'] if isinstance(pos_outputs, dict) else pos_outputs[0]
        
        if neg_labels is not None and neg_loss_scale is None:
            unwrapped_model = unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                neg_loss = self.label_smoother(neg_outputs, neg_labels, shift_labels=True)
            else:
                neg_loss = self.label_smoother(neg_outputs, neg_labels)
        else:
            neg_loss = neg_outputs['loss'] if isinstance(neg_outputs, dict) else neg_outputs[0]

        if self.sequence_parallel_size > 1:
            from swift.trainers.xtuner import reduce_xtuner_sequence_parallel_loss
            pos_loss = reduce_xtuner_sequence_parallel_loss(pos_loss, pos_labels)
            neg_loss = reduce_xtuner_sequence_parallel_loss(neg_loss, neg_labels)

        
        if pos_labels is None:
            pos_labels = inputs['pos_labels']
        if neg_labels is None:
            neg_labels = inputs['neg_labels']
        
        margin_loss = self.compute_margin_loss(pos_outputs.logits, neg_outputs.logits, pos_labels, neg_labels, self.tokenizer)

        if self.is_encoder_decoder: # tbh: do not use 
            pos_preds = pos_outputs.logits.argmax(dim=2)[..., :]
            pos_labels = pos_labels[..., :]
            neg_preds = neg_outputs.logits.argmax(dim=2)[..., :]
            neg_labels = neg_labels[..., :]
        else:
            pos_preds = pos_outputs.logits.argmax(dim=2)[..., :-1]
            pos_labels = pos_labels[..., 1:]
            neg_preds = neg_outputs.logits.argmax(dim=2)[..., :-1]
            neg_labels = neg_labels[..., 1:]

        pos_masks = pos_labels != -100
        neg_masks = neg_labels != -100
        acc_strategy = getattr(self.args, 'acc_strategy', 'token')
        pos_acc: Optional[Tensor] = None
        neg_acc: Optional[Tensor] = None

        if self.state.global_step % self.sft_args.acc_steps == 0:
            if pos_preds.shape != pos_labels.shape:
                pass
            elif acc_strategy == 'sentence':
                pos_acc_list = []
                for i, m in enumerate(pos_masks):
                    pos_acc_list.append(torch.all(pos_preds[i, m] == pos_labels[i, m]).to(torch.int64).item())
                pos_acc = torch.tensor(pos_acc_list, device=pos_preds.device).float().mean()
            else:
                if use_torchacc(): # tbh: not used
                    ta_trim_graph()
                    pos_preds = pos_preds.to('cpu')
                    pos_masks = pos_masks.to('cpu')
                    pos_labels = pos_labels.to('cpu')
                pos_acc = (torch.masked_select(pos_preds, pos_masks) == torch.masked_select(pos_labels, pos_masks)).float().mean()
            if model.training and pos_acc is not None:
                if 'pos_acc' not in self._custom_metrics:
                    self._custom_metrics['pos_acc'] = self._pos_acc
                self._custom_metrics['pos_acc'] = self._custom_metrics['pos_acc'] + pos_acc / self.args.gradient_accumulation_steps
        if self.state.global_step % self.sft_args.acc_steps == 0:
            if neg_preds.shape != neg_labels.shape:
                pass
            elif acc_strategy == 'sentence':
                neg_acc_list = []
                for i, m in enumerate(neg_masks):
                    neg_acc_list.append(torch.all(neg_preds[i, m] == neg_labels[i, m]).to(torch.int64).item())
                neg_acc = torch.tensor(neg_acc_list, device=neg_preds.device).float().mean()
            else:
                if use_torchacc(): # tbh: not used
                    ta_trim_graph()
                    neg_preds = neg_preds.to('cpu')
                    neg_masks = neg_masks.to('cpu')
                    neg_labels = neg_labels.to('cpu')
                neg_acc = (torch.masked_select(neg_preds, neg_masks) == torch.masked_select(neg_labels, neg_masks)).float().mean()
            if model.training and neg_acc is not None:
                if 'neg_acc' not in self._custom_metrics:
                    self._custom_metrics['neg_acc'] = self._neg_acc
                self._custom_metrics['neg_acc'] = self._custom_metrics['neg_acc'] + neg_acc / self.args.gradient_accumulation_steps
        total_loss = pos_loss + neg_loss + margin_loss
        
        if return_outputs:
            outputs = {
                'pos_outputs': pos_outputs,
                'neg_outputs': neg_outputs,
                'pos_loss': pos_loss,
                'neg_loss': neg_loss,
                'margin_loss': margin_loss,
                'total_loss': total_loss
            }
            return total_loss, outputs
        else:
            return total_loss

    def get_train_dataloader(self):
        if self.sequence_parallel_size > 1:
            from swift.trainers.xtuner import get_xtuner_train_dataloader
            return get_xtuner_train_dataloader(self)
        elif use_torchacc():
            if trainer.is_datasets_available():
                import datasets

            if self.train_dataset is None:
                raise ValueError('Trainer: training requires a train_dataset.')

            train_dataset = self.train_dataset
            data_collator = self.data_collator

            if trainer.is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                train_dataset = self._remove_unused_columns(train_dataset, description='training')
            else:
                data_collator = self._get_collator_with_removed_columns(data_collator, description='training')

            return ta_train_dataloader(train_dataset, data_collator, self._get_train_sampler(), self.args,
                                       self._train_batch_size)
        else:
            return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        if not use_torchacc():
            return super().get_eval_dataloader(eval_dataset)
        else:
            if trainer.is_datasets_available():
                import datasets

            if eval_dataset is None and self.eval_dataset is None:
                raise ValueError('Trainer: evaluation requires an eval_dataset.')
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            data_collator = self.data_collator

            if trainer.is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
                eval_dataset = self._remove_unused_columns(eval_dataset, description='evaluation')
            else:
                data_collator = self._get_collator_with_removed_columns(data_collator, description='evaluation')

            return ta_eval_dataloader(eval_dataset, data_collator, self._get_eval_sampler(eval_dataset), self.args)

    def get_test_dataloader(self, test_dataset):
        if not use_torchacc():
            return super().get_test_dataloader(test_dataset)
        else:
            if trainer.is_datasets_available():
                import datasets

            data_collator = self.data_collator

            if trainer.is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
                test_dataset = self._remove_unused_columns(test_dataset, description='test')
            else:
                data_collator = self._get_collator_with_removed_columns(data_collator, description='test')

            return ta_test_dataloader(test_dataset, data_collator, self._get_eval_sampler(test_dataset), self.args)


# monkey patching
trainer.DEFAULT_PROGRESS_CALLBACK = ProgressCallbackNew
trainer.DEFAULT_CALLBACKS = [DefaultFlowCallbackNew]
trainer.PrinterCallback = PrinterCallbackNew
