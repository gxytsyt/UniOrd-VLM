#!/bin/bash
export PYTHONPATH=/workspace/UniOrd-VLM/src:$PYTHONPATH
set -x

MODEL_PATH=/workspace/models/Qwen2.5-VL-7B-Instruct
OUTPUT_DIR=saves/qwen2_5vl-7b/lora/sft_standard_alldata

CUDA_VISIBLE_DEVICES=2,3,4,5 llamafactory-cli train \
    --model_name_or_path ${MODEL_PATH} \
    --image_max_pixels 262144 \
    --video_max_pixels 16384 \
    --video_maxlen 16 \
    --trust_remote_code \
    --stage sft \
    --mixed_attention \
    --do_train \
    --do_eval \
    --finetuning_type lora \
    --lora_rank 16 \
    --lora_target q_proj,k_proj,v_proj,o_proj \
    --freeze_vision_tower \
    --freeze_multi_modal_projector \
    --dataset wikihow_video_my_train_t,wikihow_video_my_train_v,wikihow_video_my_train,recipe_my_train_t,recipe_my_train_i,recipe_my_train,wikihow_my_train_t,wikihow_my_train,wikihow_my_train_i,roc_my_train,aan_my_train,arxiv_my_train,nips_my_train \
    --eval_dataset recipe_my_test \
    --template qwen2_vl \
    --media_dir /workspace \
    --cutoff_len 16384 \
    --max_samples 100000 \
    --overwrite_cache \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 8 \
    --output_dir ${OUTPUT_DIR} \
    --logging_steps 20 \
    --save_steps 100 \
    --eval_steps 500 \
    --eval_strategy steps \
    --plot_loss \
    --overwrite_output_dir \
    --save_only_model false \
    --save_total_limit 30 \
    --save_strategy steps \
    --report_to tensorboard \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --num_train_epochs 1.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 180000000 \
    --deepspeed examples/deepspeed/ds_z3_config.json \

