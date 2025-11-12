#!/bin/bash
# 多GPU分布式训练启动脚本（单机多卡或多机多卡）

set -e

echo "=========================================="
echo "RT9700 多GPU分布式GPT训练"
echo "=========================================="

# 激活虚拟环境
if [ -d ".venv" ]; then
    echo "激活虚拟环境..."
    source .venv/bin/activate
elif [ -d "gpt_train_env" ]; then
    echo "激活虚拟环境（旧版本）..."
    source gpt_train_env/bin/activate
fi

# 配置参数
MODEL_SIZE=${1:-"small"}
EPOCHS=${2:-5}
BATCH_SIZE=${3:-16}
NPROC_PER_NODE=${4:-4}  # 每个节点的GPU数量
NNODES=${5:-1}          # 节点总数
NODE_RANK=${6:-0}       # 当前节点的rank (主节点为0)
MASTER_ADDR=${7:-"localhost"}  # 主节点地址
MASTER_PORT=${8:-29500}

echo ""
echo "训练配置:"
echo "  模型大小: $MODEL_SIZE"
echo "  训练轮数: $EPOCHS"
echo "  每GPU批次: $BATCH_SIZE"
echo "  每节点GPU数: $NPROC_PER_NODE"
echo "  节点总数: $NNODES"
echo "  当前节点Rank: $NODE_RANK"
echo "  主节点地址: $MASTER_ADDR"
echo "  主节点端口: $MASTER_PORT"
echo ""
echo "有效批次大小: $((BATCH_SIZE * NPROC_PER_NODE * NNODES * 4))"
echo ""

# 设置环境变量
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # 根据实际网络接口调整
export NCCL_IB_DISABLE=1        # 如果没有InfiniBand
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# 检查GPU
echo "检查GPU状态..."
rocm-smi

# 创建目录
if [ $NODE_RANK -eq 0 ]; then
    mkdir -p output_distributed
    mkdir -p gpt_model_distributed
fi

echo ""
echo "启动分布式训练..."
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
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 4 \
    --output_dir ./output_distributed \
    --model_save_dir ./gpt_model_distributed

if [ $NODE_RANK -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "训练完成!"
    echo "=========================================="
    echo "模型保存在: ./gpt_model_distributed"
    echo "日志保存在: ./output_distributed/logs"
    echo ""
    echo "运行以下命令测试生成:"
    echo "  python3 test_generation.py --model_path ./gpt_model_distributed"
    echo ""
    echo "查看训练日志:"
    echo "  tensorboard --logdir=./output_distributed/logs"
fi
