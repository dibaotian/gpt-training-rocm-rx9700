#!/bin/bash
# Docker 跨节点 DDP 启动脚本

set -e

# 检查参数
if [ $# -lt 2 ]; then
    echo "=========================================="
    echo "Docker 跨节点 DDP 训练启动脚本"
    echo "=========================================="
    echo ""
    echo "用法: $0 <node_rank> <master_addr> [model_size] [epochs] [network_interface]"
    echo ""
    echo "参数说明:"
    echo "  node_rank         : 节点序号 (0=主节点, 1=从节点)"
    echo "  master_addr       : 主节点IP地址"
    echo "  model_size        : 模型大小 (tiny/small/medium, 默认:tiny)"
    echo "  epochs            : 训练轮数 (默认:5)"
    echo "  network_interface : 网络接口名 (默认:eth0)"
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
NETWORK_INTERFACE=${5:-"eth0"}

# 固定配置
NNODES=2
NPROC_PER_NODE=1
MASTER_PORT=29500
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

# Docker 配置
IMAGE_NAME="rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.8.0"
CONTAINER_NAME="gpt-train-node${NODE_RANK}"
SHM_SIZE="8G"

# 获取脚本所在目录（绝对路径）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 处理Docker挂载命名空间问题：/data分区可能无法被Docker访问
if [[ "$SCRIPT_DIR" == /data/min/gpt_train ]]; then
    ORIGINAL_DIR="$SCRIPT_DIR"
    SCRIPT_DIR="${HOME}/Documents/min/gpt_train"
    echo "检测到/data路径，自动转换为Docker兼容路径"
    echo "  原路径: $ORIGINAL_DIR"
    echo "  挂载路径: $SCRIPT_DIR"
fi

echo "=========================================="
echo "Docker 跨节点 DDP 训练启动"
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

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "错误: Docker未安装"
    echo "请运行: sudo apt-get install -y docker.io"
    exit 1
fi

# 检查用户是否在docker组
if ! groups | grep -q docker; then
    echo "警告: 当前用户不在docker组，使用sudo运行"
    DOCKER_CMD="sudo docker"
else
    DOCKER_CMD="docker"
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

# 检查网络连通性（从节点检查主节点）
if [ $NODE_RANK -ne 0 ]; then
    echo "检查与主节点的网络连通性..."
    if ping -c 1 -W 2 $MASTER_ADDR &> /dev/null; then
        echo "  ✓ 可以ping通主节点 $MASTER_ADDR"
    else
        echo "  ✗ 无法ping通主节点 $MASTER_ADDR"
        echo "  请检查网络连接"
        exit 1
    fi
    echo ""
fi

# 拉取镜像（如果需要）
echo "检查Docker镜像..."
if ! $DOCKER_CMD images | grep -q "rocm/pytorch.*rocm7.1"; then
    echo "正在拉取镜像 (约10GB，可能需要几分钟)..."
    $DOCKER_CMD pull $IMAGE_NAME
else
    echo "  ✓ 镜像已存在"
fi
echo ""

# 停止并删除同名容器（如果存在）
if $DOCKER_CMD ps -a | grep -q $CONTAINER_NAME; then
    echo "停止现有容器 $CONTAINER_NAME..."
    $DOCKER_CMD stop $CONTAINER_NAME 2>/dev/null || true
    $DOCKER_CMD rm $CONTAINER_NAME 2>/dev/null || true
fi

# 训练参数配置
case $MODEL_SIZE in
    tiny)
        BATCH_SIZE=16
        GRAD_ACCUM=8
        ;;
    small)
        BATCH_SIZE=8
        GRAD_ACCUM=16
        ;;
    medium)
        BATCH_SIZE=4
        GRAD_ACCUM=32
        ;;
    *)
        echo "错误: 不支持的模型大小 $MODEL_SIZE"
        echo "支持: tiny, small, medium"
        exit 1
        ;;
esac

echo "启动容器..."
echo "=========================================="
echo ""

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
  -w /workspace \
  -e MASTER_ADDR=$MASTER_ADDR \
  -e MASTER_PORT=$MASTER_PORT \
  -e RANK=$NODE_RANK \
  -e WORLD_SIZE=$WORLD_SIZE \
  -e NODE_RANK=$NODE_RANK \
  -e NCCL_SOCKET_IFNAME=$NETWORK_INTERFACE \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_DEBUG=INFO \
  -e NCCL_BUFFSIZE=2097152 \
  -e HSA_OVERRIDE_GFX_VERSION=12.0.1 \
  -e PYTORCH_ROCM_ARCH=gfx1201 \
  -e AMD_SERIALIZE_KERNEL=3 \
  -e GPU_MAX_HW_QUEUES=1 \
  -e HSA_ENABLE_SDMA=0 \
  -e HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -e PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128 \
  -e HF_ENDPOINT=https://hf-mirror.com \
  $IMAGE_NAME \
  /bin/bash -c "
    echo '=========================================='
    echo 'Docker DDP 容器已启动'
    echo '=========================================='
    echo ''
    echo '节点信息:'
    echo '  节点Rank: $NODE_RANK'
    echo '  主节点: $MASTER_ADDR:$MASTER_PORT'
    echo '  World Size: $WORLD_SIZE'
    echo ''
    echo '环境配置:'
    echo '  NCCL接口: $NCCL_SOCKET_IFNAME'
    echo '  GPU架构: $HSA_OVERRIDE_GFX_VERSION'
    echo ''
    echo '环境信息:'
    echo '  PyTorch: ' \$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '未安装')
    
    # 检查GPU
    if python3 -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
        echo '  GPU可用: ✓'
        echo '  GPU设备: ' \$(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')
        echo ''
        echo 'GPU状态:'
        rocm-smi --showid --showproductname 2>/dev/null || rocm-smi 2>/dev/null || echo '  无法获取GPU信息'
    else
        echo '  GPU可用: ✗'
        echo '  警告: PyTorch无法访问GPU'
    fi
    echo ''
    
    # 检查并自动选择网络接口
    echo '检查网络接口...'
    if ip addr show $NETWORK_INTERFACE &> /dev/null; then
        echo \"  指定接口: $NETWORK_INTERFACE\"
        ACTUAL_INTERFACE=$NETWORK_INTERFACE
    else
        echo \"  警告: 接口 $NETWORK_INTERFACE 不存在，自动选择接口\"
        # 自动选择第一个有IP的非loopback接口
        ACTUAL_INTERFACE=\$(ip -br addr show | grep -v '^lo' | grep -v 'DOWN' | head -1 | awk '{print \$1}')
        if [ -n \"\$ACTUAL_INTERFACE\" ]; then
            echo \"  自动选择: \$ACTUAL_INTERFACE\"
        else
            echo \"  错误: 找不到可用的网络接口\"
            echo \"  可用接口列表:\"
            ip -br addr show
            exit 1
        fi
    fi
    
    # 显示接口信息
    IP_ADDR=\$(ip addr show \$ACTUAL_INTERFACE | grep 'inet ' | awk '{print \$2}')
    echo \"  接口: \$ACTUAL_INTERFACE\"
    echo \"  IP地址: \$IP_ADDR\"
    echo ''
    
    # 更新NCCL环境变量
    export NCCL_SOCKET_IFNAME=\$ACTUAL_INTERFACE
    echo \"更新 NCCL_SOCKET_IFNAME=\$ACTUAL_INTERFACE\"
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
        echo ''
        echo '等待从节点连接...'
        echo '请在从节点上运行:'
        echo '  ./docker_run_ddp.sh 1 $MASTER_ADDR $MODEL_SIZE $EPOCHS $NETWORK_INTERFACE'
        echo ''
    else
        echo '📡 从节点就绪'
        echo ''
        echo '正在连接主节点 $MASTER_ADDR:$MASTER_PORT ...'
        echo ''
    fi
    
    echo '----------------------------------------'
    echo '选择操作:'
    echo '  1. 自动开始训练 (推荐)'
    echo '  2. 进入交互式shell (手动控制)'
    echo '----------------------------------------'
    echo ''
    
    # 等待用户输入或超时自动开始
    read -t 10 -p '请选择 [1/2] (10秒后自动选择1): ' choice || choice=1
    echo ''
    
    if [ \"\$choice\" = \"2\" ]; then
        echo '进入交互式模式...'
        echo ''
        echo '手动启动训练命令:'
        echo '  torchrun \\'
        echo '    --nproc_per_node=$NPROC_PER_NODE \\'
        echo '    --nnodes=$NNODES \\'
        echo '    --node_rank=$NODE_RANK \\'
        echo '    --master_addr=$MASTER_ADDR \\'
        echo '    --master_port=$MASTER_PORT \\'
        echo '    train_multi_gpu.py \\'
        echo '    --model_size $MODEL_SIZE \\'
        echo '    --use_chinese \\'
        echo '    --epochs $EPOCHS \\'
        echo '    --batch_size $BATCH_SIZE \\'
        echo '    --gradient_accumulation_steps $GRAD_ACCUM \\'
        echo '    --bf16'
        echo ''
        /bin/bash
    else
        echo '=========================================='
        echo '🚀 开始训练...'
        echo '=========================================='
        echo ''
        
        # 确保输出目录存在（仅主节点）
        if [ $NODE_RANK -eq 0 ]; then
            mkdir -p ./output_docker_ddp_${MODEL_SIZE}
            mkdir -p ./gpt_model_docker_ddp_${MODEL_SIZE}
        fi
        
        # 安装依赖（首次运行）
        echo '检查Python依赖...'
        if ! python3 -c 'import transformers' 2>/dev/null; then
            echo '首次运行，正在安装依赖包...'
            pip3 install --no-cache-dir transformers datasets accelerate tensorboard tqdm -q
            echo '✓ 依赖安装完成'
        else
            echo '✓ 依赖已安装'
        fi
        echo ''
        
        # 等待一下确保两个节点都准备好
        sleep 3
        
        # 启动训练
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
            --output_dir ./output_docker_ddp_${MODEL_SIZE} \
            --model_save_dir ./gpt_model_docker_ddp_${MODEL_SIZE}
        
        echo ''
        echo '=========================================='
        echo '✅ 训练完成！'
        echo '=========================================='
        
        if [ $NODE_RANK -eq 0 ]; then
            echo ''
            echo '模型保存: ./gpt_model_docker_ddp_${MODEL_SIZE}'
            echo '日志保存: ./output_docker_ddp_${MODEL_SIZE}/logs'
            echo ''
            echo '测试生成:'
            echo '  python3 test_generation.py \\'
            echo '    --model_path ./gpt_model_docker_ddp_${MODEL_SIZE} \\'
            echo '    --prompt \"人工智能\" \\'
            echo '    --max_length 100'
            echo ''
        fi
        
        # 训练完成后进入shell
        echo '按任意键退出容器，或继续使用shell...'
        /bin/bash
    fi
  "

echo ""
echo "容器已退出"
echo ""
