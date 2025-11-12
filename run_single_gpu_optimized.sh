#!/bin/bash

# 单GPU优化训练脚本
# 针对 GPU 利用率高但 VRAM 使用率低的场景优化

echo "=================================================="
echo "单GPU GPT训练 - 优化版"
echo "=================================================="

# 设置Python路径（如果使用虚拟环境）
# source /path/to/venv/bin/activate

# 基础配置
MODEL_SIZE="small"          # 模型大小: tiny, small, medium
DATASET="wikitext"          # 数据集
EPOCHS=3                    # 训练轮数

# 优化参数 - 根据您的 GPU 显存调整
BATCH_SIZE=32              # 批次大小 (从原来的8增加到32)
GRADIENT_ACCUM=4           # 梯度累积步数
MAX_LENGTH=512             # 序列长度

# 性能优化选项
ENABLE_FP16=""             # 启用混合精度: "--fp16"
ENABLE_GRAD_CKPT=""        # 启用梯度检查点: "--gradient_checkpointing"
NUM_WORKERS=4              # 数据加载工作进程数

# 输出目录
OUTPUT_DIR="./output_single_optimized"
MODEL_SAVE_DIR="./gpt_model_optimized"

echo ""
echo "训练配置:"
echo "  模型大小: $MODEL_SIZE"
echo "  批次大小: $BATCH_SIZE"
echo "  梯度累积: $GRADIENT_ACCUM"
echo "  有效批次: $((BATCH_SIZE * GRADIENT_ACCUM))"
echo "  混合精度: ${ENABLE_FP16:-禁用}"
echo "  梯度检查点: ${ENABLE_GRAD_CKPT:-禁用}"
echo ""
echo "预期 VRAM 使用率: 50-80%"
echo "=================================================="
echo ""

# 运行训练
python3 train_single_gpu_optimized.py \
    --model_size $MODEL_SIZE \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUM \
    --max_length $MAX_LENGTH \
    --dataloader_num_workers $NUM_WORKERS \
    --output_dir $OUTPUT_DIR \
    --model_save_dir $MODEL_SAVE_DIR \
    $ENABLE_FP16 \
    $ENABLE_GRAD_CKPT

echo ""
echo "=================================================="
echo "训练完成!"
echo "=================================================="
echo ""
echo "查看训练日志:"
echo "  tensorboard --logdir=$OUTPUT_DIR/logs"
echo ""
echo "进一步优化建议:"
echo "  1. 如果 VRAM 使用率 < 60%, 可以增大 BATCH_SIZE"
echo "  2. 如果遇到 OOM, 可以:"
echo "     - 减小 BATCH_SIZE"
echo "     - 增大 GRADIENT_ACCUM"
echo "     - 启用 --fp16 (修改脚本中的 ENABLE_FP16)"
echo "     - 启用 --gradient_checkpointing"
echo "  3. 监控 GPU 使用率:"
echo "     watch -n 1 rocm-smi"
echo ""
