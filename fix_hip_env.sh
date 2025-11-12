#!/bin/bash
# 设置正确的ROCm/HIP环境变量

echo "=========================================="
echo "配置ROCm/HIP环境变量"
echo "=========================================="
echo ""

# 查询GFX版本
echo "查询GPU架构..."
GFX_VERSION=$(rocminfo 2>/dev/null | grep "Name:" | grep "gfx" | head -1 | awk '{print $2}')

if [ -z "$GFX_VERSION" ]; then
    echo "⚠️  无法自动检测GFX版本"
    echo "请手动设置:"
    echo "  export HSA_OVERRIDE_GFX_VERSION=11.0.0  # 或 11.0.1"
    echo "  export PYTORCH_ROCM_ARCH=gfx1100        # 或 gfx1101"
    echo ""
    # 使用默认值
    GFX_VERSION="gfx1100"
    echo "使用默认值: $GFX_VERSION"
else
    echo "✓ 检测到GFX: $GFX_VERSION"
fi

# 根据GFX版本设置环境变量
case $GFX_VERSION in
    gfx1201)
        export HSA_OVERRIDE_GFX_VERSION=12.0.1
        export PYTORCH_ROCM_ARCH=gfx1201
        echo "检测到 RDNA4 架构 (AMD Radeon AI PRO R9700)"
        ;;
    gfx1200)
        export HSA_OVERRIDE_GFX_VERSION=12.0.0
        export PYTORCH_ROCM_ARCH=gfx1200
        ;;
    gfx1101)
        export HSA_OVERRIDE_GFX_VERSION=11.0.1
        export PYTORCH_ROCM_ARCH=gfx1101
        ;;
    gfx1100)
        export HSA_OVERRIDE_GFX_VERSION=11.0.0
        export PYTORCH_ROCM_ARCH=gfx1100
        ;;
    gfx1030)
        export HSA_OVERRIDE_GFX_VERSION=10.3.0
        export PYTORCH_ROCM_ARCH=gfx1030
        ;;
    *)
        echo "未知GFX版本: $GFX_VERSION，使用默认值"
        export HSA_OVERRIDE_GFX_VERSION=12.0.1
        export PYTORCH_ROCM_ARCH=gfx1201
        ;;
esac

# 调试和优化设置（注意：AMD_SERIALIZE_KERNEL只能是0或1）
export AMD_SERIALIZE_KERNEL=1
export AMD_LOG_LEVEL=3
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export GPU_MAX_HW_QUEUES=4
export HSA_ENABLE_SDMA=0

echo ""
echo "环境变量已设置:"
echo "  HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
echo "  PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH"
echo "  AMD_SERIALIZE_KERNEL=$AMD_SERIALIZE_KERNEL"
echo ""

# 测试GPU
echo "测试GPU基本操作..."
python3 << 'EOF'
import torch
try:
    print(f"PyTorch版本: {torch.__version__}")
    print(f"GPU可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        # 测试简单张量操作
        x = torch.randn(100, 100).cuda()
        y = x + x
        print("✓ 张量操作成功")
        # 测试简单模型
        model = torch.nn.Linear(100, 100).cuda()
        out = model(x)
        print("✓ 模型操作成功")
        print("\n✅ GPU环境配置正确！")
    else:
        print("❌ GPU不可用")
except Exception as e:
    print(f"❌ 测试失败: {e}")
    print("\n请尝试不同的HSA_OVERRIDE_GFX_VERSION值:")
    print("  export HSA_OVERRIDE_GFX_VERSION=11.0.1")
    print("  或")
    print("  export HSA_OVERRIDE_GFX_VERSION=10.3.0")
EOF

echo ""
echo "=========================================="
echo "现在可以运行训练:"
echo "  python3 train_single_gpu.py --model_size tiny"
echo ""
echo "如果仍有错误，尝试手动设置:"
echo "  export HSA_OVERRIDE_GFX_VERSION=11.0.1"
echo "  export PYTORCH_ROCM_ARCH=gfx1101"
echo "=========================================="
