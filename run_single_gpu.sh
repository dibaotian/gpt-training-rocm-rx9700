#!/bin/bash
# 单GPU训练启动脚本

set -e  # 遇到错误立即退出

echo "=================================="
echo "RT9700 单GPU GPT训练"
echo "=================================="

# 激活虚拟环境（如果存在）
if [ -d ".venv" ]; then
    echo "激活虚拟环境..."
    source .venv/bin/activate
elif [ -d "gpt_train_env" ]; then
    echo "激活虚拟环境（旧版本）..."
    source gpt_train_env/bin/activate
fi

# 检查GPU
echo "检查GPU状态..."
rocm-smi || echo "警告: rocm-smi不可用"

# 设置环境变量
# export PYTORCH_ROCM_ARCH=gfx1201  # 根据您的GPU调整,已经在docker 运行的时候设置

# 默认参数（可以通过命令行覆盖）
MODEL_SIZE=${1:-"tiny"}  # tiny, small, medium
EPOCHS=${2:-3}
BATCH_SIZE=${3:-8}

echo ""
echo "训练参数:"
echo "  模型大小: $MODEL_SIZE"
echo "  训练轮数: $EPOCHS"
echo "  批次大小: $BATCH_SIZE"
echo ""

# 创建必要的目录
mkdir -p output_single
mkdir -p gpt_model

# 运行训练
python3 train_single_gpu.py \
    --model_size $MODEL_SIZE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --output_dir ./output_single \
    --model_save_dir ./gpt_model

echo ""
echo "=================================="
echo "训练完成!"
echo "=================================="
echo "模型保存在: ./gpt_model"
echo "日志保存在: ./output_single/logs"
echo ""
echo "运行以下命令测试生成:"
echo "  python3 test_generation.py --model_path ./gpt_model"
echo ""
echo "查看训练日志:"
echo "  tensorboard --logdir=./output_single/logs"
