#!/bin/bash
export PYTHONPATH=/workspace/UniOrd-VLM/src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2,3,4,5

# ========================================
MODEL_PATH=/workspace/models/Qwen2.5-VL-7B-Instruct
LORA_BASE_DIR=saves/qwen2_5vl-7b/lora/sft_standard_alldata
CHECKPOINTS=(200)


RESULTS_SUMMARY="${LORA_BASE_DIR}/all_checkpoints_summary_recipe.txt"
echo "$(date)" >> ${RESULTS_SUMMARY}
echo "==========================================" >> ${RESULTS_SUMMARY}
echo "" >> ${RESULTS_SUMMARY}

for CKPT in "${CHECKPOINTS[@]}"; do
    echo ""
    echo "=================================================="
    echo "📍 [$(date +%H:%M:%S)] Checkpoint-${CKPT}"
    echo "=================================================="

    ADAPTER_PATH=${LORA_BASE_DIR}/checkpoint-${CKPT}
    OUTPUT_DIR='${LORA_BASE_DIR}/recipe/checkpoint-${CKPT}'

    echo "ADAPTER_PATH: ${ADAPTER_PATH}"
    echo "OUTPUT_DIR: ${OUTPUT_DIR}"
    echo ""
    # ========================================

    llamafactory-cli train \
        --model_name_or_path ${MODEL_PATH} \
        --adapter_name_or_path ${ADAPTER_PATH} \
        --video_max_pixels 16384 \
        --video_maxlen 16 \
        --stage sft \
        --do_predict \
        --mixed_attention \
        --finetuning_type lora \
        --predict_with_generate \
        --eval_dataset recipe_my_test \
        --template qwen2_vl \
        --media_dir /workspace \
        --cutoff_len 8192 \
        --max_samples 10000 \
        --preprocessing_num_workers 16 \
        --output_dir ${OUTPUT_DIR} \
        --overwrite_output_dir \
        --overwrite_cache \
        --max_new_tokens 512 \
        --temperature 0.1 \
        --top_p 0.9 \
        --do_sample \
        --per_device_eval_batch_size 1 \
        --dataloader_num_workers 4 \
        --trust_remote_code \
        --image_max_pixels 262144 \
        --bf16

    if [ $? -ne 0 ]; then
        echo "Checkpoint-${CKPT}: ❌ Failed" >> ${RESULTS_SUMMARY}
        echo "" >> ${RESULTS_SUMMARY}
        continue
    fi

    echo "✅"

    # ========================================
    echo ""
    echo "🔍..."

    python examples/test_my/cal_test_result.py \
        --input ${OUTPUT_DIR}/generated_predictions.jsonl \
        --output ${OUTPUT_DIR}/metrics_results.json

    if [ $? -ne 0 ]; then
        echo "Checkpoint-${CKPT}: ⚠️  Failed" >> ${RESULTS_SUMMARY}
        echo "" >> ${RESULTS_SUMMARY}
        continue
    fi

    # ========================================
    echo "✅"

    # 读取指标结果
    if [ -f "${OUTPUT_DIR}/metrics_results.json" ]; then
        ACC=$(python3 -c "import json; print(json.load(open('${OUTPUT_DIR}/metrics_results.json'))['acc'])")
        PMR=$(python3 -c "import json; print(json.load(open('${OUTPUT_DIR}/metrics_results.json'))['pmr'])")
        TAU=$(python3 -c "import json; print(json.load(open('${OUTPUT_DIR}/metrics_results.json'))['taus'])")
        PM=$(python3 -c "import json; print(json.load(open('${OUTPUT_DIR}/metrics_results.json'))['pm'])")

        echo ""
        echo "📊 Checkpoint-${CKPT} 结果:"
        echo "  ACC:  ${ACC}"
        echo "  PMR:  ${PMR}"
        echo "  Tau:  ${TAU}"
        echo "  PM:   ${PM}"

        echo "Checkpoint-${CKPT}: ✅ 成功" >> ${RESULTS_SUMMARY}
        echo "  ACC:  ${ACC}" >> ${RESULTS_SUMMARY}
        echo "  PMR:  ${PMR}" >> ${RESULTS_SUMMARY}
        echo "  Tau:  ${TAU}" >> ${RESULTS_SUMMARY}
        echo "  PM:   ${PM}" >> ${RESULTS_SUMMARY}
        echo "" >> ${RESULTS_SUMMARY}

        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "⚠️"
        echo "Checkpoint-${CKPT}: ⚠️" >> ${RESULTS_SUMMARY}
        echo "" >> ${RESULTS_SUMMARY}
    fi

    echo "=================================================="
    echo "✅ Checkpoint-${CKPT}"
    echo "=================================================="

done

