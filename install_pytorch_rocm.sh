#!/bin/bash
# PyTorch ROCm 安装脚本
# 由于 PyTorch 官方还没有提供 ROCm 7.1 的预编译包，
# 我们使用 ROCm 6.2 版本（与 ROCm 7.1 向后兼容）

set -e

echo "=== PyTorch ROCm 安装脚本 ==="
echo ""
echo "检测到的 ROCm 版本："
rocm-smi --showproductname 2>/dev/null | grep "gfx" || echo "无法检测 GPU"
echo ""

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo "错误: 未找到虚拟环境，请先运行: uv venv"
    exit 1
fi

echo "步骤 1: 卸载已安装的 PyTorch（如果存在）..."
uv pip uninstall torch torchvision torchaudio triton 2>/dev/null || true

echo ""
echo "步骤 2: 安装 PyTorch ROCm 6.2 版本..."
echo "注意: ROCm 6.2 的 PyTorch 与 ROCm 7.1 兼容"
echo ""

# 使用 pip 模式安装（通过 uv）
# PyTorch 官方支持的最新 ROCm 版本是 6.2
uv pip install \
    --index https://download.pytorch.org/whl/rocm6.2 \
    torch torchvision torchaudio

echo ""
echo "步骤 3: 验证安装..."
.venv/bin/python << 'EOF'
import torch
print(f"✓ PyTorch 版本: {torch.__version__}")
print(f"✓ ROCm 支持: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")
print(f"✓ CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ 检测到 GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("⚠ 警告: PyTorch 无法检测到 GPU")
    print("  这可能是正常的，取决于你的系统配置")
EOF

echo ""
echo "=== 安装完成 ==="
echo ""
echo "如果 PyTorch 无法检测到 GPU，请检查:"
echo "  1. ROCm 驱动是否正确安装"
echo "  2. 环境变量是否设置正确"
echo "  3. 运行: rocm-smi 查看 GPU 状态"
