#!/bin/bash
# 中文GPT模型训练快速启动脚本

set -e

echo "=========================================="
echo "中文GPT模型训练"
echo "=========================================="

# 参数配置
MODEL_SIZE=${1:-"small"}  # 模型大小：tiny, small, medium, large, xl
EPOCHS=${2:-5}            # 训练轮数
DATA_PERCENT=${3:-5}      # 数据集百分比

# 根据模型大小自动调整批次大小
case $MODEL_SIZE in
    tiny)
        BATCH_SIZE=32
        GRAD_ACCUM=1
        ;;
    small)
        BATCH_SIZE=32
        GRAD_ACCUM=8
        ;;
    medium)
        BATCH_SIZE=8
        GRAD_ACCUM=8
        ;;
    large)
        BATCH_SIZE=4
        GRAD_ACCUM=16
        ;;
    xl)
        BATCH_SIZE=2
        GRAD_ACCUM=32
        ;;
    *)
        BATCH_SIZE=1
        GRAD_ACCUM=32
        ;;
esac

echo ""
echo "训练配置:"
echo "  模型大小: $MODEL_SIZE"
echo "  训练轮数: $EPOCHS"
echo "  数据集比例: ${DATA_PERCENT}%"
echo "  批次大小: $BATCH_SIZE"
echo "  梯度累积: $GRAD_ACCUM"
echo "  有效批次: $((BATCH_SIZE * GRAD_ACCUM))"
echo ""

# 设置Hugging Face镜像（加速下载）
export HF_ENDPOINT=https://hf-mirror.com

# 激活虚拟环境（如果存在）
if [ -d ".venv" ]; then
    echo "激活虚拟环境..."
    source .venv/bin/activate
elif [ -d "gpt_train_env" ]; then
    echo "激活虚拟环境..."
    source gpt_train_env/bin/activate
fi

# 检查GPU
echo "检查GPU状态..."
if command -v rocm-smi &> /dev/null; then
    rocm-smi | head -20
fi

# 输出和模型目录
OUTPUT_DIR="./output_chinese_${MODEL_SIZE}"
MODEL_DIR="./gpt_model_chinese_${MODEL_SIZE}"

echo ""
echo "输出目录: $OUTPUT_DIR"
echo "模型保存: $MODEL_DIR"
echo ""

# 创建目录
mkdir -p $OUTPUT_DIR
mkdir -p $MODEL_DIR

# 运行训练
echo "开始训练..."
echo "=========================================="

# 使用中文维基百科训练GPT-2 Small
# python3 train_single_gpu.py \
#     --model_size small \
#     --use_chinese \
#     --epochs 5 \
#     --batch_size 16 \
#     --gradient_accumulation_steps 4 \
#     --fp16 \
#     --output_dir ./output_chinese_small \
#     --model_save_dir ./gpt_model_chinese_small


python3 train_single_gpu.py \
    --model_size $MODEL_SIZE \
    --use_chinese \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_length 512 \
    --learning_rate 5e-5 \
    --bf16 \
    --output_dir $OUTPUT_DIR \
    --model_save_dir $MODEL_DIR

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "模型保存在: $MODEL_DIR"
echo "日志保存在: $OUTPUT_DIR/logs"
echo ""
echo "测试生成:"
echo "  python3 test_generation.py \\"
echo "    --model_path $MODEL_DIR \\"
echo "    --prompt \"从前有一座山，\" \\"
echo "    --max_length 100"
echo ""
echo "查看训练日志:"
echo "  tensorboard --logdir=$OUTPUT_DIR/logs"
echo ""
