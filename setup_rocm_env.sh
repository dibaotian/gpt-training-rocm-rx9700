#!/bin/bash
# ROCm 环境设置脚本
# 设置必要的环境变量以便 PyTorch 能够检测到 AMD GPU

# ROCm 安装路径
export ROCM_PATH=/opt/rocm
export ROCM_HOME=/opt/rocm

# 将 ROCm 添加到 PATH
export PATH=$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH

# 设置库路径
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# HIP 相关设置
export HIP_PATH=$ROCM_PATH
export HIP_PLATFORM=amd

# 设置设备可见性（默认显示所有 GPU）
export ROCR_VISIBLE_DEVICES=0
export GPU_DEVICE_ORDINAL=0
export HIP_VISIBLE_DEVICES=0

# HSA 设置
export HSA_OVERRIDE_GFX_VERSION=12.0.1

# RCCL 设置（用于多 GPU 训练）
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

echo "✓ ROCm 环境变量已设置"
echo "  ROCM_PATH: $ROCM_PATH"
echo "  HIP_PLATFORM: $HIP_PLATFORM"
echo "  HSA_OVERRIDE_GFX_VERSION: $HSA_OVERRIDE_GFX_VERSION"
