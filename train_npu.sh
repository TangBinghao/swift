NNODES=2 \
NODE_RANK=0 \
MASTER_ADDR=29.35.222.244 \
NPROC_PER_NODE=8 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 swift sft  \
    --ddp_find_unused_parameters true \
    --sft_type full \
    --model_type internvl-v2   \
    --output_dir /mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/swift_new/output/ \
    --model_cache_dir /mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/data/InternVL2-2B \
    --custom_train_dataset_path /mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/swift-main/person_train_mllm_swift_v2.jsonl \
    --custom_val_dataset_path /mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/swift-main/person_test_mllm_swift_v2.jsonl \
    --save_strategy epoch \
    --save_total_limit -1 \
    --num_train_epochs 8 \
    --max_length 6000 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules DEFAULT \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.01 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 6000 \
    --save_steps 6000 \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --dtype bf16 \
    --ddp_backend=hccl \
    --save_on_each_node false \


NNODES=2 \
NODE_RANK=1 \
MASTER_ADDR=29.35.222.244 \
NPROC_PER_NODE=8 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 swift sft  \
    --ddp_find_unused_parameters true \
    --sft_type full \
    --model_type internvl-v2   \
    --output_dir /mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/swift_new/output/ \
    --model_cache_dir /mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/data/InternVL2-2B \
    --custom_train_dataset_path /mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/swift-main/person_train_mllm_swift_v2.jsonl \
    --custom_val_dataset_path /mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/swift-main/person_test_mllm_swift_v2.jsonl \
    --save_strategy epoch \
    --save_total_limit -1 \
    --num_train_epochs 8 \
    --max_length 6000 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules DEFAULT \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.01 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 6000 \
    --save_steps 6000 \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --dtype bf16 \
    --ddp_backend=hccl \
    --save_on_each_node false \
    # --freeze_parameters 1 \
    # --additional_trainable_parameters language_model mlp1 \