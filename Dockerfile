# Dockerfile for GPT Training with ROCm PyTorch
# 基于官方 ROCm PyTorch 镜像，预装训练所需的所有依赖

FROM rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.8.0

# 设置工作目录
WORKDIR /workspace

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    PIP_NO_CACHE_DIR=1

# 更新系统并安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    iproute2 \
    net-tools \
    iputils-ping \
    telnet \
    htop \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖包
RUN pip3 install --no-cache-dir \
    transformers \
    datasets \
    accelerate \
    tensorboard \
    tqdm \
    sentencepiece \
    protobuf \
    wandb

# 创建工作目录结构
RUN mkdir -p /workspace/output \
    /workspace/models \
    /workspace/datasets_cache

# 设置 GPU 环境变量（针对 RDNA4 gfx1201）
ENV HSA_OVERRIDE_GFX_VERSION=12.0.1 \
    PYTORCH_ROCM_ARCH=gfx1201 \
    AMD_SERIALIZE_KERNEL=3 \
    GPU_MAX_HW_QUEUES=1 \
    HSA_ENABLE_SDMA=0 \
    HSA_FORCE_FINE_GRAIN_PCIE=1 \
    PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128

# 设置 NCCL/RCCL 环境变量
ENV NCCL_IB_DISABLE=1 \
    NCCL_DEBUG=INFO \
    NCCL_BUFFSIZE=2097152

# 验证安装
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" && \
    python3 -c "import datasets; print(f'Datasets: {datasets.__version__}')"

# 设置默认命令
CMD ["/bin/bash"]
