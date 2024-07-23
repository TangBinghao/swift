#!/bin/bash

# 基本参数
export NNODES=2
export NODE_RANK=1
export NPROC_PER_NODE=8
export MASTER_ADDR=29.35.167.21
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
OUTPUT_DIR="/mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/swift_pair/output_pretrain/"
MODEL_CACHE_DIR="/mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/data/InternVL2-2B"
VAL_DATASET_PATH="/mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/swift_pair/person_train_mllm_swift_pairwise.jsonl"
NUM_EPOCHS=1
BATCH_SIZE=4
LEARNING_RATE=1e-5
DDP_BACKEND="hccl"
DEEPSPEED_CONFIG="default-zero3"

# 切分后的训练数据文件前缀
TRAIN_DATA_PREFIX="/mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/data/pretrain/split_file_part_"

# 获取切分后的文件数量
NUM_PARTS=$(ls ${TRAIN_DATA_PREFIX}*.jsonl | wc -l)

# 逐个文件进行训练
for ((i=1; i<=NUM_PARTS; i++))
do
    TRAIN_DATASET_PATH="${TRAIN_DATA_PREFIX}${i}.jsonl"
    
    # 查找上一轮保存的检查点
    if [ $i -eq 1 ]; then
        RESUME_FROM_CHECKPOINT="None"
    else
        PREV_OUTPUT_DIR="${OUTPUT_DIR_BASE}$((i-1))"
        RESUME_FROM_CHECKPOINT=$(ls -t ${PREV_OUTPUT_DIR}/checkpoint-* | head -n 1)
        if [ -z "$RESUME_FROM_CHECKPOINT" ]; then
            RESUME_FROM_CHECKPOINT="None"
        fi
    fi


    # 打印当前训练的信息
    echo "Training with dataset: ${TRAIN_DATASET_PATH}"
    echo "Resuming from checkpoint: ${RESUME_FROM_CHECKPOINT}"
    # 运行训练脚本
    if [ "$RESUME_FROM_CHECKPOINT" == "None" ]; then
        swift sft \
            --ddp_find_unused_parameters true \
            --sft_type full \
            --model_type internvl2-2b \
            --output_dir ${OUTPUT_DIR}${i} \
            --model_cache_dir ${MODEL_CACHE_DIR} \
            --custom_train_dataset_path ${TRAIN_DATASET_PATH} \
            --custom_val_dataset_path ${VAL_DATASET_PATH} \
            --add_output_dir_suffix false \
            --save_strategy epoch \
            --save_total_limit -1 \
            --num_train_epochs ${NUM_EPOCHS} \
            --max_length 8192 \
            --lora_rank 8 \
            --lora_alpha 32 \
            --lora_dropout_p 0.05 \
            --lora_target_modules DEFAULT \
            --gradient_checkpointing true \
            --batch_size ${BATCH_SIZE} \
            --weight_decay 0.01 \
            --learning_rate ${LEARNING_RATE} \
            --lazy_tokenize true \
            --preprocess_num_proc 8 \
            --gradient_accumulation_steps 1 \
            --eval_steps 100000 \
            --save_steps 100000 \
            --max_grad_norm 0.5 \
            --warmup_ratio 0.03 \
            --dtype bf16 \
            --ddp_backend=${DDP_BACKEND} \
            --save_on_each_node false \
            --deepspeed ${DEEPSPEED_CONFIG} \
            --freeze_parameters 1 \
            --additional_trainable_parameters language_model mlp1 \
    else
        swift sft \
            --ddp_find_unused_parameters true \
            --sft_type full \
            --model_type internvl2-2b \
            --output_dir ${OUTPUT_DIR}_${i} \
            --model_cache_dir ${MODEL_CACHE_DIR} \
            --custom_train_dataset_path ${TRAIN_DATASET_PATH} \
            --custom_val_dataset_path ${VAL_DATASET_PATH} \
            --add_output_dir_suffix false \
            --save_strategy epoch \
            --save_total_limit -1 \
            --num_train_epochs ${NUM_EPOCHS} \
            --max_length 8192 \
            --lora_rank 8 \
            --lora_alpha 32 \
            --lora_dropout_p 0.05 \
            --lora_target_modules DEFAULT \
            --gradient_checkpointing true \
            --batch_size ${BATCH_SIZE} \
            --weight_decay 0.01 \
            --learning_rate ${LEARNING_RATE} \
            --lazy_tokenize true \
            --preprocess_num_proc 8 \
            --gradient_accumulation_steps 1 \
            --eval_steps 100000 \
            --save_steps 100000 \
            --max_grad_norm 0.5 \
            --warmup_ratio 0.03 \
            --dtype bf16 \
            --ddp_backend=${DDP_BACKEND} \
            --save_on_each_node false \
            --deepspeed ${DEEPSPEED_CONFIG} \
            --freeze_parameters 1 \
            --additional_trainable_parameters language_model mlp1 \
            --resume_from_checkpoint ${RESUME_FROM_CHECKPOINT}
    fi
done