#!/bin/bash
# Docker 镜像构建脚本

set -e

echo "=========================================="
echo "构建自定义 GPT 训练 Docker 镜像"
echo "=========================================="
echo ""

# 配置
IMAGE_NAME="gpt-train-rocm"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "错误: Docker 未安装"
    echo "请运行: sudo apt-get install -y docker.io"
    exit 1
fi

# 检查 Dockerfile
if [ ! -f "Dockerfile" ]; then
    echo "错误: 找不到 Dockerfile"
    echo "请确保在 gpt_train 目录下运行此脚本"
    exit 1
fi

echo "镜像配置:"
echo "  名称: $FULL_IMAGE_NAME"
echo "  基础镜像: rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.8.0"
echo ""

# 显示 Dockerfile 内容摘要
echo "将安装以下依赖:"
echo "  - transformers (Hugging Face)"
echo "  - datasets"
echo "  - accelerate"
echo "  - tensorboard"
echo "  - tqdm"
echo "  - sentencepiece"
echo "  - protobuf"
echo "  - wandb (Weights & Biases)"
echo ""

echo "将配置以下环境变量:"
echo "  - HSA_OVERRIDE_GFX_VERSION=12.0.1"
echo "  - PYTORCH_ROCM_ARCH=gfx1201"
echo "  - NCCL_IB_DISABLE=1"
echo "  - HF_ENDPOINT=https://hf-mirror.com"
echo ""

read -p "开始构建镜像？ [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "=========================================="
echo "开始构建..."
echo "=========================================="
echo ""

# 构建镜像
if groups | grep -q docker; then
    DOCKER_CMD="docker"
else
    echo "使用 sudo 运行 docker..."
    DOCKER_CMD="sudo docker"
fi

# 开始构建
BUILD_START=$(date +%s)

$DOCKER_CMD build \
    --tag $FULL_IMAGE_NAME \
    .

BUILD_END=$(date +%s)
BUILD_TIME=$((BUILD_END - BUILD_START))

echo ""
echo "=========================================="
echo "✅ 镜像构建完成！"
echo "=========================================="
echo ""
echo "构建时间: ${BUILD_TIME}秒"
echo "镜像名称: $FULL_IMAGE_NAME"
echo ""

# 显示镜像信息
echo "镜像信息:"
$DOCKER_CMD images $IMAGE_NAME

echo ""
echo "=========================================="
echo "下一步"
echo "=========================================="
echo ""
echo "1. 测试镜像:"
echo "   docker run -it --rm $FULL_IMAGE_NAME python3 -c 'import torch; print(torch.__version__)'"
echo ""
echo "2. 使用新镜像启动 DDP 训练:"
echo "   ./docker_run_ddp_custom.sh 0 <主节点IP>"
echo ""
echo "3. 推送到 Docker Hub (可选):"
echo "   docker tag $FULL_IMAGE_NAME <your-username>/$IMAGE_NAME:$IMAGE_TAG"
echo "   docker push <your-username>/$IMAGE_NAME:$IMAGE_TAG"
echo ""
echo "4. 在其他节点拉取镜像:"
echo "   docker pull <your-username>/$IMAGE_NAME:$IMAGE_TAG"
echo ""
