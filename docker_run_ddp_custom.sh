#!/bin/bash
# Docker 跨节点 DDP 启动脚本（使用自定义镜像）

set -e

# 检查参数
if [ $# -lt 2 ]; then
    echo "=========================================="
    echo "Docker 跨节点 DDP 训练启动脚本"
    echo "（使用自定义镜像 gpt-train-rocm:latest）"
    echo "=========================================="
    echo ""
    echo "用法: $0 <node_rank> <master_addr> [model_size] [epochs] [network_interface]"
    echo ""
    echo "参数说明:"
    echo "  node_rank         : 节点序号 (0=主节点, 1=从节点)"
    echo "  master_addr       : 主节点IP地址"
    echo "  model_size        : 模型大小 (tiny/small/medium, 默认:tiny)"
    echo "  epochs            : 训练轮数 (默认:5)"
    echo "  network_interface : 网络接口名 (默认:auto，自动检测)"
    echo ""
    echo "示例:"
    echo "  主节点: $0 0 192.168.1.100"
    echo "  从节点: $0 1 192.168.1.100"
    echo ""
    echo "  自定义: $0 0 192.168.1.100 small 10 eno1"
    echo ""
    exit 1
fi

NODE_RANK=$1
MASTER_ADDR=$2
MODEL_SIZE=${3:-"tiny"}
EPOCHS=${4:-5}
NETWORK_INTERFACE=${5:-"auto"}

# 固定配置
NNODES=2
NPROC_PER_NODE=1
MASTER_PORT=29500
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

# Docker 配置 - 使用自定义镜像
IMAGE_NAME="gpt-train-rocm:latest"
CONTAINER_NAME="gpt-train-custom-node${NODE_RANK}"
SHM_SIZE="8G"

# 获取脚本所在目录（绝对路径）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 处理Docker挂载命名空间问题
if [[ "$SCRIPT_DIR" == /data/min/gpt_train ]]; then
    ORIGINAL_DIR="$SCRIPT_DIR"
    SCRIPT_DIR="${HOME}/Documents/min/gpt_train"
    echo "检测到/data路径，自动转换为Docker兼容路径"
    echo "  原路径: $ORIGINAL_DIR"
    echo "  挂载路径: $SCRIPT_DIR"
fi

echo "=========================================="
echo "Docker 跨节点 DDP 训练启动"
echo "（使用自定义镜像）"
echo "=========================================="
echo ""
echo "节点配置:"
echo "  节点Rank: $NODE_RANK ($([ $NODE_RANK -eq 0 ] && echo '主节点' || echo '从节点'))"
echo "  总节点数: $NNODES"
echo "  主节点地址: $MASTER_ADDR"
echo "  主节点端口: $MASTER_PORT"
echo "  World Size: $WORLD_SIZE"
echo ""
echo "训练配置:"
echo "  模型大小: $MODEL_SIZE"
echo "  训练轮数: $EPOCHS"
echo "  网络接口: $NETWORK_INTERFACE"
echo ""
echo "Docker 配置:"
echo "  镜像: $IMAGE_NAME"
echo "  容器名: $CONTAINER_NAME"
echo "  工作目录: $SCRIPT_DIR"
echo "  网络模式: host"
echo ""

# 检查Docker
if ! command -v docker &> /dev/null; then
    echo "错误: Docker未安装"
    exit 1
fi

# 检查自定义镜像是否存在
if groups | grep -q docker; then
    DOCKER_CMD="docker"
else
    DOCKER_CMD="sudo docker"
fi

if ! $DOCKER_CMD images | grep -q "gpt-train-rocm"; then
    echo "错误: 自定义镜像 $IMAGE_NAME 不存在"
    echo ""
    echo "请先构建镜像:"
    echo "  cd /path/to/gpt_train"
    echo "  ./build_docker_image.sh"
    echo ""
    exit 1
fi

# 检查GPU设备
echo "检查GPU设备..."
if [ ! -e /dev/kfd ] || [ ! -e /dev/dri ]; then
    echo "警告: 未找到AMD GPU设备"
    echo "  /dev/kfd: $([ -e /dev/kfd ] && echo '✓' || echo '✗')"
    echo "  /dev/dri: $([ -e /dev/dri ] && echo '✓' || echo '✗')"
else
    echo "  ✓ GPU设备正常"
fi
echo ""

# 检查网络连通性（从节点）
if [ $NODE_RANK -ne 0 ]; then
    echo "检查与主节点的网络连通性..."
    if ping -c 1 -W 2 $MASTER_ADDR &> /dev/null; then
        echo "  ✓ 可以ping通主节点 $MASTER_ADDR"
    else
        echo "  ✗ 无法ping通主节点 $MASTER_ADDR"
        exit 1
    fi
    echo ""
fi

# 停止并删除同名容器
if $DOCKER_CMD ps -a | grep -q $CONTAINER_NAME; then
    echo "停止现有容器 $CONTAINER_NAME..."
    $DOCKER_CMD stop $CONTAINER_NAME 2>/dev/null || true
    $DOCKER_CMD rm $CONTAINER_NAME 2>/dev/null || true
fi

# 训练参数配置
case $MODEL_SIZE in
    tiny)
        BATCH_SIZE=32
        GRAD_ACCUM=4
        ;;
    small)
        BATCH_SIZE=32
        GRAD_ACCUM=4
        ;;
    medium)
        BATCH_SIZE=4
        GRAD_ACCUM=32
        ;;
    *)
        echo "错误: 不支持的模型大小 $MODEL_SIZE"
        exit 1
        ;;
esac

echo "启动容器..."
echo "=========================================="
echo ""

# 准备 WandB 配置挂载（如果存在）
WANDB_MOUNT_ARGS=""
if [ -f "$HOME/.netrc" ]; then
    WANDB_MOUNT_ARGS="$WANDB_MOUNT_ARGS -v $HOME/.netrc:/root/.netrc:ro"
fi
if [ -d "$HOME/.config/wandb" ]; then
    WANDB_MOUNT_ARGS="$WANDB_MOUNT_ARGS -v $HOME/.config/wandb:/root/.config/wandb"
fi

# 启动容器
$DOCKER_CMD run -it --rm \
  --name $CONTAINER_NAME \
  --network host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --shm-size $SHM_SIZE \
  --mount type=bind,source="$SCRIPT_DIR",target=/workspace,bind-propagation=rslave \
  $WANDB_MOUNT_ARGS \
  -w /workspace \
  -e MASTER_ADDR=$MASTER_ADDR \
  -e MASTER_PORT=$MASTER_PORT \
  -e RANK=$NODE_RANK \
  -e WORLD_SIZE=$WORLD_SIZE \
  -e NODE_RANK=$NODE_RANK \
  -e NCCL_SOCKET_IFNAME=$NETWORK_INTERFACE \
  -e WANDB_API_KEY=${WANDB_API_KEY:-} \
  $IMAGE_NAME \
  /bin/bash -c "
    echo '=========================================='
    echo 'Docker DDP 容器已启动（自定义镜像）'
    echo '=========================================='
    echo ''
    echo '节点信息:'
    echo '  节点Rank: $NODE_RANK'
    echo '  主节点: $MASTER_ADDR:$MASTER_PORT'
    echo '  World Size: $WORLD_SIZE'
    echo ''
    
    # 检查GPU
    if python3 -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
        echo 'GPU状态:'
        echo '  ✓ GPU可用'
        echo '  设备: ' \$(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')
        rocm-smi --showid --showproductname 2>/dev/null | head -5 || true
    else
        echo '  ✗ GPU不可用'
    fi
    echo ''
    
    # 检查并安装 iproute2（如果缺失）
    if ! command -v ip &> /dev/null; then
        echo '检测到缺少 ip 命令，正在安装 iproute2...'
        apt-get update -qq && apt-get install -y -qq iproute2 > /dev/null 2>&1
        echo '  ✓ iproute2 已安装'
    fi
    
    # 自动检测网络接口
    echo '检查网络接口...'
    if [ \"$NETWORK_INTERFACE\" = \"auto\" ]; then
        ACTUAL_INTERFACE=\$(ip -br addr show 2>/dev/null | grep -v '^lo' | grep -v 'DOWN' | head -1 | awk '{print \$1}')
        if [ -z \"\$ACTUAL_INTERFACE\" ]; then
            # 降级使用 ifconfig
            ACTUAL_INTERFACE=\$(ifconfig | grep -E '^[a-z]' | grep -v '^lo' | head -1 | awk '{print \$1}' | tr -d ':')
        fi
        echo \"  自动检测: \$ACTUAL_INTERFACE\"
    else
        if ip addr show $NETWORK_INTERFACE &> /dev/null; then
            ACTUAL_INTERFACE=$NETWORK_INTERFACE
            echo \"  使用指定接口: \$ACTUAL_INTERFACE\"
        else
            ACTUAL_INTERFACE=\$(ip -br addr show 2>/dev/null | grep -v '^lo' | grep -v 'DOWN' | head -1 | awk '{print \$1}')
            if [ -z \"\$ACTUAL_INTERFACE\" ]; then
                ACTUAL_INTERFACE=\$(ifconfig | grep -E '^[a-z]' | grep -v '^lo' | head -1 | awk '{print \$1}' | tr -d ':')
            fi
            echo \"  接口 $NETWORK_INTERFACE 不存在，自动选择: \$ACTUAL_INTERFACE\"
        fi
    fi
    
    IP_ADDR=\$(ip addr show \$ACTUAL_INTERFACE 2>/dev/null | grep 'inet ' | awk '{print \$2}')
    if [ -z \"\$IP_ADDR\" ]; then
        IP_ADDR=\$(ifconfig \$ACTUAL_INTERFACE | grep 'inet ' | awk '{print \$2}')
    fi
    echo \"  接口: \$ACTUAL_INTERFACE\"
    echo \"  IP地址: \$IP_ADDR\"
    export NCCL_SOCKET_IFNAME=\$ACTUAL_INTERFACE
    echo ''
    
    echo '=========================================='
    echo '训练配置:'
    echo '  模型: $MODEL_SIZE'
    echo '  轮数: $EPOCHS'
    echo '  批次: $BATCH_SIZE'
    echo '  梯度累积: $GRAD_ACCUM'
    echo '  有效批次: \$((BATCH_SIZE * WORLD_SIZE * GRAD_ACCUM))'
    echo '=========================================='
    echo ''
    
    if [ $NODE_RANK -eq 0 ]; then
        echo '🚀 主节点就绪'
        echo '等待从节点连接...'
    else
        echo '📡 从节点就绪'
        echo '连接主节点 $MASTER_ADDR:$MASTER_PORT ...'
    fi
    echo ''
    
    # 自动开始训练（3秒后）
    echo '3秒后自动开始训练...'
    sleep 3
    
    echo '=========================================='
    echo '🚀 开始训练...'
    echo '=========================================='
    echo ''
    
    # 确保输出目录存在
    if [ $NODE_RANK -eq 0 ]; then
        mkdir -p ./output_custom_ddp_${MODEL_SIZE}
        mkdir -p ./gpt_model_custom_ddp_${MODEL_SIZE}
    fi
    
    # 启动训练（依赖已预装在镜像中）
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
        --output_dir ./output_custom_ddp_${MODEL_SIZE} \
        --model_save_dir ./gpt_model_custom_ddp_${MODEL_SIZE}
    
    echo ''
    echo '=========================================='
    echo '✅ 训练完成！'
    echo '=========================================='
    
    if [ $NODE_RANK -eq 0 ]; then
        echo ''
        echo '模型保存: ./gpt_model_custom_ddp_${MODEL_SIZE}'
        echo '日志保存: ./output_custom_ddp_${MODEL_SIZE}/logs'
        echo ''
    fi
    
    echo '按任意键退出容器...'
    /bin/bash
  "

echo ""
echo "容器已退出"
echo ""
