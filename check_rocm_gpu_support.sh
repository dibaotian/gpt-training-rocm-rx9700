#!/bin/bash
# ROCm GPU 检查脚本
# 整合了 gfx1201 和 HIP gfx12 支持检查

echo "=========================================="
echo "ROCm GPU 完整支持检查"
echo "=========================================="
echo ""

# 1. 系统信息
echo "1. 系统信息"
echo "-------------------"
echo "Ubuntu 版本:"
cat /etc/os-release | grep VERSION_ID

echo ""
echo "内核版本:"
uname -r

echo ""
echo "ROCm 版本:"
if [ -f /opt/rocm/.info/version ]; then
    cat /opt/rocm/.info/version
else
    echo "未找到版本文件"
fi

echo ""
echo "HIP 版本:"
hipconfig --version 2>/dev/null || echo "hipconfig 不可用"
echo ""

# 2. PyTorch 信息
echo "2. PyTorch 信息"
echo "-------------------"
python3 << 'EOF'
import torch
import sys

print(f"PyTorch 版本: {torch.__version__}")

if hasattr(torch.version, 'hip'):
    print(f"HIP 版本: {torch.version.hip}")
else:
    print("警告: 无 HIP 版本信息")

print(f"GPU 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    props = torch.cuda.get_device_properties(0)
    print(f"总显存: {props.total_memory / 1024**3:.2f} GB")
EOF
echo ""

# 3. GPU 架构信息
echo "3. GPU 架构信息"
echo "-------------------"
if command -v rocminfo &> /dev/null; then
    echo "实际 GPU 架构:"
    rocminfo | grep "Name:" | grep "gfx" | head -3
else
    echo "rocminfo 不可用"
fi
echo ""

# 4. PyTorch 支持的架构
echo "4. PyTorch 编译时包含的架构"
echo "-------------------"
python3 << 'EOF'
import torch
import sys

try:
    archs = torch.cuda.get_arch_list()
    print("PyTorch 支持的架构列表:")
    for arch in archs:
        marker = ""
        if 'gfx1201' in str(arch):
            marker = " ✅ (原生支持 gfx1201)"
        elif 'gfx12' in str(arch):
            marker = " ✅ (gfx12xx 系列)"
        elif 'gfx1101' in str(arch):
            marker = " ⚠️  (兼容模式候选)"
        print(f"  - {arch}{marker}")
    
    # 分析支持情况
    has_gfx1201 = any('gfx1201' in str(arch) for arch in archs)
    has_gfx12 = any('gfx12' in str(arch) for arch in archs)
    has_gfx1101 = any('gfx1101' in str(arch) for arch in archs)
    
    print("")
    if has_gfx1201:
        print("✅ 检测到 gfx1201 原生支持")
    elif has_gfx12:
        print("✅ 检测到 gfx12xx 系列支持（可能支持 gfx1201）")
    elif has_gfx1101:
        print("⚠️  检测到 gfx1101 支持（可通过兼容模式运行 gfx1201）")
        
except Exception as e:
    print(f"检查失败: {e}")
    sys.exit(1)
EOF
echo ""

# 5. ROCm 库文件检查
echo "5. ROCm 库文件检查"
echo "-------------------"
echo "查找 gfx12 相关文件 (前15个结果):"
find /opt/rocm -name "*gfx12*" 2>/dev/null | head -15 || echo "未找到 gfx12 相关文件"
echo ""

echo "检查 HIP 运行时库中的 gfx12 符号:"
if [ -f /opt/rocm/lib/libamdhip64.so ]; then
    strings /opt/rocm/lib/libamdhip64.so | grep -i "gfx12" | head -10 || echo "未找到 gfx12 符号"
else
    echo "libamdhip64.so 未找到"
fi
echo ""

# 6. PyTorch 库检查
echo "6. PyTorch 库中的架构支持"
echo "-------------------"
TORCH_LIB=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))" 2>/dev/null)
if [ -n "$TORCH_LIB" ]; then
    echo "PyTorch 安装路径: $TORCH_LIB"
    echo "检查 .so 文件中的 gfx12 符号 (前5个结果):"
    find "$TORCH_LIB" -name "*.so" -exec sh -c 'strings "$1" 2>/dev/null | grep -q "gfx12" && echo "  ✓ $1 包含 gfx12"' _ {} \; 2>/dev/null | head -5 || echo "  未找到包含 gfx12 的库文件"
else
    echo "无法定位 PyTorch 安装路径"
fi
echo ""

# 7. 环境变量检查
echo "7. 当前环境变量"
echo "-------------------"
echo "ROCm/HIP/GPU 相关环境变量:"
env | grep -E "(HSA|HIP|ROCM|AMD|GPU|PYTORCH)" | sort
echo ""

# 8. 编译器支持检查
echo "8. ROCm 编译器支持"
echo "-------------------"
if command -v amdclang++ &> /dev/null; then
    echo "检查编译器支持的 gfx12 目标:"
    amdclang++ --help 2>&1 | grep -i "gfx12" | head -5 || echo "未找到 gfx12 支持信息"
else
    echo "amdclang++ 未找到"
fi
echo ""

# 9. 总结和建议
echo "=========================================="
echo "总结和建议"
echo "=========================================="
echo ""

python3 << 'EOF'
import torch
import sys

try:
    archs = torch.cuda.get_arch_list()
    has_gfx1201 = any('gfx1201' in str(arch) for arch in archs)
    has_gfx12 = any('gfx12' in str(arch) for arch in archs)
    has_gfx1101 = any('gfx1101' in str(arch) for arch in archs)
    
    print("=" * 60)
    
    if has_gfx1201:
        print("✅ 当前环境支持 gfx1201 原生模式")
        print("")
        print("推荐配置 (docker_run.sh):")
        print("  export HSA_OVERRIDE_GFX_VERSION=12.0.1")
        print("  export PYTORCH_ROCM_ARCH=gfx1201")
        print("  export AMD_SERIALIZE_KERNEL=3")
        print("  export GPU_MAX_HW_QUEUES=1")
        print("")
        print("优点:")
        print("  - 原生性能，无性能损失")
        print("  - 充分利用 RDNA4 架构特性")
        
    elif has_gfx12:
        print("✅ 当前环境支持 gfx12xx 系列")
        print("")
        print("建议:")
        print("  1. 优先尝试 gfx1201 原生配置")
        print("  2. 如果失败，使用 gfx1101 兼容模式")
        print("")
        print("测试原生模式:")
        print("  export HSA_OVERRIDE_GFX_VERSION=12.0.1")
        print("  export PYTORCH_ROCM_ARCH=gfx1201")
        
    else:
        print("⚠️  PyTorch 不包含 gfx1201 原生支持，使用兼容模式")
        print("")
        print("当前解决方案 (已在 docker_run.sh 中配置):")
        print("  export HSA_OVERRIDE_GFX_VERSION=12.0.1")
        print("  export PYTORCH_ROCM_ARCH=gfx1201")
        print("")
        print("说明:")
        print("  - 虽然 PyTorch 编译时未包含 gfx1201")
        print("  - 但通过 HSA_OVERRIDE 可以运行")
        print("  - 功能完全可用")
        print("")
        print("长期解决方案:")
        print("  1. 等待更新的 ROCm/PyTorch 镜像")
        print("  2. 从源码编译 PyTorch with gfx1201 支持")
    
    print("")
    print("=" * 60)
    
except Exception as e:
    print(f"分析失败: {e}")
    sys.exit(1)
EOF

echo ""
echo "=========================================="
echo "检查完成"
echo "=========================================="
