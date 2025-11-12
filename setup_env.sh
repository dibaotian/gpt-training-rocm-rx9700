#!/bin/bash
# GPT训练环境自动设置脚本（使用uv）

set -e

echo "=========================================="
echo "GPT训练环境设置 - RT9700 ROCm (使用uv)"
echo "=========================================="
echo ""

# 检查Python版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "检测到Python版本: $PYTHON_VERSION"

# 检查是否满足最低版本要求 (Python 3.8+)
if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "错误: 需要Python 3.8或更高版本，当前版本: $PYTHON_VERSION"
    exit 1
fi

echo "✓ Python版本检查通过"

# 检查uv是否已安装
if ! command -v uv &> /dev/null; then
    echo ""
    echo "uv未安装，正在安装uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # uv可能安装在不同位置，尝试添加到PATH
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    
    # 如果安装后仍然找不到，尝试source环境脚本
    if [ -f "$HOME/.local/bin/env" ]; then
        source "$HOME/.local/bin/env"
    fi
    
    # 再次检查
    if ! command -v uv &> /dev/null; then
        echo "错误: uv安装失败或无法找到"
        echo "请手动运行: source $HOME/.local/bin/env"
        echo "然后重新运行此脚本"
        exit 1
    fi
    
    echo "✓ uv安装完成: $(uv --version)"
else
    echo ""
    echo "检测到uv: $(uv --version)"
fi

# 使用uv创建虚拟环境
if [ ! -d ".venv" ]; then
    echo ""
    echo "使用uv创建虚拟环境..."
    uv venv
    echo "✓ 虚拟环境创建完成"
else
    echo ""
    echo "虚拟环境已存在，跳过创建"
fi

# 激活虚拟环境
echo ""
echo "激活虚拟环境..."
source .venv/bin/activate

# 检测ROCm版本
echo ""
echo "检测ROCm版本..."

# 使用dpkg检测ROCm版本（更可靠）
if dpkg -l 2>/dev/null | grep -q "^ii.*rocm-core"; then
    ROCM_VERSION=$(dpkg -l | grep "^ii.*rocm-core" | awk '{print $3}' | cut -d. -f1,2)
    echo "检测到ROCm版本: $ROCM_VERSION (通过dpkg)"
    
    # 根据ROCm版本选择PyTorch
    ROCM_MAJOR=$(echo $ROCM_VERSION | cut -d. -f1)
    ROCM_MINOR=$(echo $ROCM_VERSION | cut -d. -f2)
    
    if [ "$ROCM_MAJOR" -eq 7 ]; then
        echo "⚠ 注意: ROCm 7.x是较新版本"
        echo "   尝试使用PyTorch rocm6.2索引（如果可用）"
        TORCH_INDEX="rocm6.2"
        echo "   如果安装失败，将回退到rocm6.1"
    elif [ "$ROCM_MAJOR" -eq 6 ]; then
        if [ "$ROCM_MINOR" -ge 2 ]; then
            TORCH_INDEX="rocm6.2"
        elif [ "$ROCM_MINOR" -ge 1 ]; then
            TORCH_INDEX="rocm6.1"
        else
            TORCH_INDEX="rocm6.0"
        fi
    elif [ "$ROCM_MAJOR" -eq 5 ]; then
        TORCH_INDEX="rocm5.7"
    else
        echo "警告: 未识别的ROCm版本，默认使用rocm6.1"
        TORCH_INDEX="rocm6.1"
    fi
else
    echo "警告: 未检测到ROCm安装，默认使用rocm6.1"
    TORCH_INDEX="rocm6.1"
fi
echo "将使用PyTorch索引: $TORCH_INDEX"

# 使用uv安装PyTorch
echo ""
echo "使用uv安装PyTorch (ROCm版本: $TORCH_INDEX)..."

# 尝试安装，如果失败则回退
if ! uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$TORCH_INDEX; then
    if [[ $TORCH_INDEX == "rocm6.2" ]]; then
        echo ""
        echo "⚠ rocm6.2安装失败，回退到rocm6.1..."
        TORCH_INDEX="rocm6.1"
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$TORCH_INDEX
    else
        echo ""
        echo "错误: PyTorch安装失败"
        echo "请访问 https://pytorch.org/get-started/locally/ 查看支持的ROCm版本"
        exit 1
    fi
fi

echo "✓ PyTorch安装完成"

# 使用uv安装其他依赖
echo ""
echo "使用uv安装其他依赖..."
uv pip install -r requirements.txt

# 验证安装
echo ""
echo "=========================================="
echo "验证安装"
echo "=========================================="

# 检查PyTorch
echo ""
echo "检查PyTorch..."
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}')"

# # 检查GPU
# echo ""
# echo "检查GPU可用性..."
# echo "注意: PyTorch for ROCm使用torch.cuda API（兼容CUDA接口）"
# python3 -c "import torch; print(f'GPU可用: {torch.cuda.is_available()}')"

# if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
#     python3 -c "import torch; print(f'GPU设备: {torch.cuda.get_device_name(0)}')"
#     python3 -c "import torch; print(f'GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')"
#     python3 -c "import torch; print(f'计算能力: {\".\".join(map(str, torch.cuda.get_device_capability(0)))}')"
#     echo ""
#     echo "✓ GPU检测成功! (ROCm后端)"
# else
#     echo ""
#     echo "⚠ 警告: GPU不可用"
#     echo ""
#     echo "可能的原因:"
#     echo "  1. ROCm驱动未正确安装或未加载"
#     echo "  2. 用户不在render/video组 (需要: sudo usermod -a -G render,video \$USER)"
#     echo "  3. 环境变量未设置 (需要: export HSA_OVERRIDE_GFX_VERSION=11.0.0)"
#     echo "  4. /dev/kfd设备节点权限问题"
#     echo ""
#     echo "诊断命令:"
#     echo "  ls -la /dev/kfd /dev/dri/"
#     echo "  lsmod | grep amdgpu"
#     echo "  groups"
#     echo "  rocm-smi"
#     echo ""
#     echo "详细排错请查看: cat SETUP_GUIDE.md"
# fi

# 检查transformers
echo ""
echo "检查Transformers..."
python3 -c "import transformers; print(f'Transformers版本: {transformers.__version__}')"

echo ""
echo "=========================================="
echo "环境设置完成!"
echo "=========================================="
echo ""
echo "下一步:"
echo "  1. 激活环境: source .venv/bin/activate"
echo "  2. 或使用uv运行: uv run python train_single_gpu.py"
echo "  3. 运行训练: ./run_single_gpu.sh"
echo "  4. 查看README: cat README.md"
echo ""
echo "uv常用命令:"
echo "  uv pip install <package>  - 安装包"
echo "  uv pip list               - 列出已安装的包"
echo "  uv run <command>          - 在虚拟环境中运行命令"
echo ""
