#!/bin/bash
# Docker容器启动脚本 - 用于GPT训练

set -e

echo "=========================================="
echo "启动ROCm PyTorch Docker容器"
echo "=========================================="
echo ""

# 配置
# IMAGE_NAME="rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.8.0"
IMAGE_NAME="gpt-train-rocm:latest"
CONTAINER_NAME="gpt-train-rocm"
SHM_SIZE="8G"

# 获取脚本所在目录（绝对路径）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "脚本所在目录: $SCRIPT_DIR"

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "错误: Docker未安装"
    echo "请运行以下命令安装:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install -y docker.io"
    exit 1
fi

# 检查用户是否在docker组
if ! groups | grep -q docker; then
    echo "警告: 当前用户不在docker组"
    echo "建议运行: sudo usermod -a -G docker \$USER"
    echo "然后重新登录"
    echo ""
    echo "继续使用sudo运行..."
    DOCKER_CMD="sudo docker"
else
    DOCKER_CMD="docker"
fi

# 检查GPU设备
echo "检查GPU设备..."
if [ ! -e /dev/kfd ] || [ ! -e /dev/dri ]; then
    echo "警告: 未找到AMD GPU设备"
    echo "  /dev/kfd: $([ -e /dev/kfd ] && echo '存在' || echo '不存在')"
    echo "  /dev/dri: $([ -e /dev/dri ] && echo '存在' || echo '不存在')"
    echo ""
    read -p "是否继续？ (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 拉取镜像（如果本地不存在）
echo ""
echo "检查Docker镜像..."
if ! $DOCKER_CMD images | grep -q "rocm/pytorch.*rocm7.1"; then
    echo "本地未找到镜像，正在拉取..."
    echo "镜像大小约 10GB，可能需要几分钟..."
    $DOCKER_CMD pull $IMAGE_NAME
else
    echo "✓ 镜像已存在"
fi

# 检查 WandB 配置
WANDB_ENV=""
if [ -f "$SCRIPT_DIR/wandb_key" ]; then
    echo ""
    echo "检测到 WandB 配置文件"
    WANDB_API_KEY=$(cat "$SCRIPT_DIR/wandb_key" | tr -d '[:space:]')
    if [ -n "$WANDB_API_KEY" ]; then
        WANDB_ENV="-e WANDB_API_KEY=$WANDB_API_KEY"
        echo "✓ WandB API Key 已加载"
    fi
fi

# 停止并删除同名容器（如果存在）
if $DOCKER_CMD ps -a | grep -q $CONTAINER_NAME; then
    echo "停止现有容器..."
    $DOCKER_CMD stop $CONTAINER_NAME 2>/dev/null || true
    $DOCKER_CMD rm $CONTAINER_NAME 2>/dev/null || true
fi

# 启动容器
echo ""
echo "启动容器..."
echo "容器名称: $CONTAINER_NAME"
echo "共享内存: $SHM_SIZE"
echo "工作目录: /workspace (挂载自 $SCRIPT_DIR)"
echo ""

$DOCKER_CMD run -it --rm \
  --name $CONTAINER_NAME \
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
  -p 6006:6006 \
  -e HSA_OVERRIDE_GFX_VERSION=12.0.1 \
  -e PYTORCH_ROCM_ARCH=gfx1201 \
  -e AMD_SERIALIZE_KERNEL=3 \
  -e GPU_MAX_HW_QUEUES=1 \
  -e HSA_ENABLE_SDMA=0 \
  -e HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -e PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128 \
  $WANDB_ENV \
  $IMAGE_NAME \
  /bin/bash -c "
    echo '=========================================='
    echo 'Docker容器已启动'
    echo '=========================================='
    echo ''
    echo '环境配置:'
    echo '  HSA_OVERRIDE_GFX_VERSION=\$HSA_OVERRIDE_GFX_VERSION'
    echo '  PYTORCH_ROCM_ARCH=\$PYTORCH_ROCM_ARCH'
    echo ''
    echo '环境信息:'
    echo '  PyTorch版本:' \$(python3 -c 'import torch; print(torch.__version__)')
    echo '  GPU可用:' \$(python3 -c 'import torch; print(torch.cuda.is_available())')
    if python3 -c 'import torch; exit(0 if torch.cuda.is_available() else 1)'; then
        echo '  GPU设备:' \$(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')
        echo ''
        echo 'GPU状态:'
        rocm-smi
        echo ''
        echo '查询GFX架构:'
        rocminfo | grep Name: | grep  gfx 
    fi
    echo ''
    echo '=========================================='
    echo '首次使用需要安装依赖:'
    echo '  pip3 install -r requirements.txt'
    echo ''
    echo '开始训练:'
    echo '  基础训练（不使用WandB）:'
    echo '    python3 train_single_gpu.py --model_size tiny'
    echo ''
    echo '  使用 WandB 追踪训练（推荐，可实时查看指标）:'
    echo '    python3 train_single_gpu.py --model_size tiny --wandb_project gpt-training --wandb_run_name tiny-test'
    echo ''
    echo '测试生成:'
    echo '  python3 test_generation.py'
    echo ''
    echo 'WandB 相关参数说明:'
    echo '  --wandb_project <项目名>    指定 WandB 项目名称'
    echo '  --wandb_run_name <运行名>   指定本次训练的运行名称'
    echo '  --wandb_entity <团队名>     指定 WandB 团队/用户名（可选）'
    echo ''
    echo '查看 TensorBoard（在新终端中运行）:'
    echo '  tensorboard --logdir=./output_single/logs --bind_all'
    echo '  然后访问: http://<宿主机IP>:6006'
    echo '=========================================='
    echo ''
    /bin/bash
  "

echo ""
echo "容器已退出"
