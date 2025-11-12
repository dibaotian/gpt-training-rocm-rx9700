#!/bin/bash
# ROCm GPU 诊断脚本
# 用于排查 HSA_STATUS_ERROR_EXCEPTION 等 ROCm 错误

echo "=========================================="
echo "ROCm GPU 环境诊断工具"
echo "=========================================="
echo ""

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# 1. 检查 ROCm 安装
echo "1. 检查 ROCm 安装"
echo "-------------------"
if command -v rocminfo &> /dev/null; then
    print_success "rocminfo 可用"
    ROCM_VERSION=$(rocminfo | grep "Runtime Version" | head -1 || echo "未知")
    echo "   ROCm 版本: $ROCM_VERSION"
else
    print_error "rocminfo 未找到"
fi

if command -v rocm-smi &> /dev/null; then
    print_success "rocm-smi 可用"
else
    print_error "rocm-smi 未找到"
fi
echo ""

# 2. 检查 GPU 设备
echo "2. 检查 GPU 设备"
echo "-------------------"
if [ -e /dev/kfd ]; then
    print_success "/dev/kfd 存在"
    ls -l /dev/kfd
else
    print_error "/dev/kfd 不存在"
fi

if [ -e /dev/dri ]; then
    print_success "/dev/dri 存在"
    ls -l /dev/dri/ | head -5
else
    print_error "/dev/dri 不存在"
fi
echo ""

# 3. 检查 GPU 信息
echo "3. 检查 GPU 信息"
echo "-------------------"
if command -v rocminfo &> /dev/null; then
    echo "GPU 架构信息:"
    GFX_VERSION=$(rocminfo 2>/dev/null | grep "Name:" | grep "gfx" | head -1 | awk '{print $2}')
    if [ -n "$GFX_VERSION" ]; then
        print_success "检测到 GFX 版本: $GFX_VERSION"
    else
        print_error "无法检测 GFX 版本"
    fi
    
    echo ""
    echo "完整 GPU 信息:"
    rocminfo | grep -A 10 "Agent 2"
fi
echo ""

# 4. 检查当前环境变量
echo "4. 检查环境变量"
echo "-------------------"
env_vars=(
    "HSA_OVERRIDE_GFX_VERSION"
    "PYTORCH_ROCM_ARCH"
    "AMD_SERIALIZE_KERNEL"
    "GPU_MAX_HW_QUEUES"
    "HSA_ENABLE_SDMA"
    "ROCR_VISIBLE_DEVICES"
    "HIP_VISIBLE_DEVICES"
    "AMD_LOG_LEVEL"
    "PYTORCH_HIP_ALLOC_CONF"
)

for var in "${env_vars[@]}"; do
    value="${!var}"
    if [ -n "$value" ]; then
        echo "  $var=$value"
    else
        echo "  $var=<未设置>"
    fi
done
echo ""

# 5. 测试 PyTorch GPU
echo "5. 测试 PyTorch GPU"
echo "-------------------"
python3 << 'EOF'
import sys
try:
    import torch
    print(f"PyTorch 版本: {torch.__version__}")
    
    if hasattr(torch.version, 'hip'):
        print(f"HIP 版本: {torch.version.hip}")
    else:
        print("警告: 无法获取 HIP 版本")
    
    print(f"CUDA API 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU 数量: {torch.cuda.device_count()}")
        print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
        
        # 获取设备属性
        props = torch.cuda.get_device_properties(0)
        print(f"总显存: {props.total_memory / 1024**3:.2f} GB")
        print(f"计算能力: {props.major}.{props.minor}")
    else:
        print("错误: GPU 不可用")
        sys.exit(1)
        
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    print_success "PyTorch GPU 基本检查通过"
else
    print_error "PyTorch GPU 基本检查失败"
fi
echo ""

# 6. 测试简单 GPU 操作
echo "6. 测试简单 GPU 操作"
echo "-------------------"
python3 << 'EOF'
import sys
import os

# 设置更详细的日志
os.environ['AMD_LOG_LEVEL'] = '4'

try:
    import torch
    
    print("测试 1: 创建小张量...")
    x = torch.randn(10, 10, device='cuda')
    print("  ✓ 成功")
    
    print("测试 2: 张量加法...")
    y = x + x
    print("  ✓ 成功")
    
    print("测试 3: 矩阵乘法（小规模）...")
    z = torch.matmul(x, x)
    print("  ✓ 成功")
    
    print("测试 4: 创建较大张量...")
    large_x = torch.randn(1000, 1000, device='cuda')
    print("  ✓ 成功")
    
    print("测试 5: 大规模矩阵乘法...")
    large_z = torch.matmul(large_x, large_x)
    torch.cuda.synchronize()
    print("  ✓ 成功")
    
    print("\n✓ 所有基本操作测试通过")
    
except Exception as e:
    print(f"\n✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

TEST_RESULT=$?
echo ""

# 7. 建议
echo "=========================================="
echo "诊断总结与建议"
echo "=========================================="
echo ""

if [ $TEST_RESULT -eq 0 ]; then
    print_success "GPU 环境正常"
    echo ""
    echo "可以开始训练:"
    echo "  python3 train_single_gpu.py --model_size tiny"
else
    print_error "GPU 环境存在问题"
    echo ""
    echo "建议尝试以下修复步骤:"
    echo ""
    echo "方案 1: 调整 GFX 版本设置"
    echo "----------------------------"
    echo "尝试不同的 HSA_OVERRIDE_GFX_VERSION 值:"
    echo "  export HSA_OVERRIDE_GFX_VERSION=11.0.1"
    echo "  export PYTORCH_ROCM_ARCH=gfx1101"
    echo ""
    echo "或者:"
    echo "  export HSA_OVERRIDE_GFX_VERSION=11.0.0"
    echo "  export PYTORCH_ROCM_ARCH=gfx1100"
    echo ""
    echo "或者:"
    echo "  export HSA_OVERRIDE_GFX_VERSION=10.3.0"
    echo "  export PYTORCH_ROCM_ARCH=gfx1030"
    echo ""
    
    echo "方案 2: 调整性能设置"
    echo "----------------------------"
    echo "  export AMD_SERIALIZE_KERNEL=3"
    echo "  export GPU_MAX_HW_QUEUES=1"
    echo "  export HSA_ENABLE_SDMA=0"
    echo "  export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128"
    echo ""
    
    echo "方案 3: 使用修复脚本"
    echo "----------------------------"
    echo "  source fix_hip_env.sh"
    echo ""
    
    echo "方案 4: 重启 Docker 容器"
    echo "----------------------------"
    echo "  exit  # 退出容器"
    echo "  ./docker_run.sh  # 重新启动"
    echo ""
    
    echo "方案 5: 检查宿主机驱动"
    echo "----------------------------"
    echo "在宿主机上运行:"
    echo "  rocm-smi"
    echo "  dmesg | grep -i amdgpu"
    echo "  ls -la /dev/kfd /dev/dri"
fi

echo ""
echo "=========================================="
