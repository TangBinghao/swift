NNODES=2 \
NODE_RANK=0 \
MASTER_ADDR=29.35.222.231 \
NPROC_PER_NODE=8 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 swift sft  \
    --ddp_find_unused_parameters true \
    --sft_type lora \
    --model_type internvl2-2b   \
    --output_dir /mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/swift_v1/output \
    --model_cache_dir /mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/data/InternVL2-2B \
    --custom_train_dataset_path /mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/swift-main/person_train_mllm_swift_v2.jsonl \
    --custom_val_dataset_path /mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/swift-main/person_test_mllm_swift_v2.jsonl \
    --save_strategy epoch \
    --save_total_limit -1 \
    --num_train_epochs 5 \
    --max_length 8192 \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.01 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100000 \
    --save_steps 100000 \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --dtype bf16 \
    --ddp_backend=hccl \
    --save_on_each_node false \
    --deepspeed default-zero3 \


NNODES=2 \
NODE_RANK=1 \
MASTER_ADDR=29.35.222.231 \
NPROC_PER_NODE=8 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 swift sft  \
    --ddp_find_unused_parameters true \
    --sft_type lora \
    --model_type internvl2-2b   \
    --output_dir /mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/swift_v1/output \
    --model_cache_dir /mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/data/InternVL2-2B \
    --custom_train_dataset_path /mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/swift-main/person_train_mllm_swift_v2.jsonl \
    --custom_val_dataset_path /mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/swift-main/person_test_mllm_swift_v2.jsonl \
    --save_strategy epoch \
    --save_total_limit -1 \
    --num_train_epochs 5 \
    --max_length 8192 \
    --batch_size 1 \
    --weight_decay 0.01 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100000 \
    --save_steps 100000 \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --dtype bf16 \
    --ddp_backend=hccl \
    --save_on_each_node false \
    --deepspeed default-zero3 \
    # --freeze_parameters 1 \
    # --additional_trainable_parameters language_model mlp1 \

