#!/bin/bash
# 双节点GPT训练启动脚本（针对1Gbps网络优化）

set -e

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <node_rank> <master_addr> [model_size] [epochs]"
    echo ""
    echo "示例:"
    echo "  主节点: $0 0 192.168.1.100"
    echo "  从节点: $0 1 192.168.1.100"
    echo ""
    echo "  自定义: $0 0 192.168.1.100 small 5"
    exit 1
fi

NODE_RANK=$1
MASTER_ADDR=$2
MODEL_SIZE=${3:-"tiny"}   # 默认tiny
EPOCHS=${4:-5}            # 默认5轮

# 固定配置（针对1Gbps网络优化）
NNODES=2
NPROC_PER_NODE=1
MASTER_PORT=29500
BATCH_SIZE=16
GRAD_ACCUM=8  # 关键：减少通信频率

echo "=========================================="
echo "双节点分布式GPT训练（1Gbps网络优化）"
echo "=========================================="
echo ""
echo "节点配置:"
echo "  当前节点Rank: $NODE_RANK"
echo "  总节点数: $NNODES"
echo "  主节点地址: $MASTER_ADDR"
echo "  主节点端口: $MASTER_PORT"
echo ""
echo "训练配置:"
echo "  模型大小: $MODEL_SIZE"
echo "  训练轮数: $EPOCHS"
echo "  每GPU批次: $BATCH_SIZE"
echo "  梯度累积: $GRAD_ACCUM 步 ⭐关键优化"
echo "  混合精度: BF16 ✅ (更稳定)"
echo "  有效批次: $((BATCH_SIZE * NNODES * GRAD_ACCUM))"
echo ""
echo "网络优化:"
echo "  梯度累积减少通信频率"
echo "  BF16减少50%传输量且数值更稳定"
echo "  预期通信占比: ~20%"
echo ""

# 设置环境变量
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # 根据实际网卡调整
export NCCL_IB_DISABLE=1
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export HF_ENDPOINT=https://hf-mirror.com  # Hugging Face镜像

# 激活虚拟环境
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "gpt_train_env" ]; then
    source gpt_train_env/bin/activate
fi

# 检查GPU
if [ $NODE_RANK -eq 0 ]; then
    echo "检查GPU状态..."
    rocm-smi | head -20 || true
    echo ""
fi

# 输出目录
OUTPUT_DIR="./output_2nodes_${MODEL_SIZE}"
MODEL_DIR="./gpt_model_2nodes_${MODEL_SIZE}"

if [ $NODE_RANK -eq 0 ]; then
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
fi

echo "启动训练..."
echo "=========================================="
echo ""

# 使用torchrun启动
torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_multi_gpu.py \
    --model_size $MODEL_SIZE \
    --use_chinese \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_length 512 \
    --bf16 \
    --output_dir $OUTPUT_DIR \
    --model_save_dir $MODEL_DIR

if [ $NODE_RANK -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "训练完成！"
    echo "=========================================="
    echo "模型保存: $MODEL_DIR"
    echo "日志保存: $OUTPUT_DIR/logs"
    echo ""
    echo "测试生成:"
    echo "  python3 test_generation.py \\"
    echo "    --model_path $MODEL_DIR \\"
    echo "    --prompt \"人工智能\" \\"
    echo "    --max_length 100"
    echo ""
    echo "查看日志:"
    echo "  tensorboard --logdir=$OUTPUT_DIR/logs"
    echo ""
fi
