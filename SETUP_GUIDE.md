# 环境设置指南和故障排除

## 📋 问题清单

根据您的运行结果，发现了以下问题：

### 1. ✅ 已解决的问题
- Python版本检查：通过 ✓
- uv安装：成功 ✓
- PyTorch安装：成功 ✓
- Transformers安装：成功 ✓

### 2. ⚠️ 需要解决的问题
- **uv在脚本外无法使用**：PATH未永久添加
- **ROCm版本检测失败**：返回空值
- **GPU不可用**：PyTorch无法识别GPU

---

## 🔧 解决方案

### 问题1: uv命令不可用

#### 原因
uv被安装在`~/.local/bin/uv`，但这个路径不在您的shell的PATH中。

#### 解决方案

**方法一：临时使用（每次重新打开终端都需要）**
```bash
export PATH="$HOME/.local/bin:$PATH"
uv --version  # 验证
```

**方法二：永久添加（推荐）**
```bash
# 对于bash用户
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 对于zsh用户
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# 验证
uv --version
```

**方法三：使用绝对路径**
```bash
~/.local/bin/uv --version
~/.local/bin/uv venv
~/.local/bin/uv pip install <package>
```

### 问题2: ROCm版本检测失败

#### 诊断步骤

```bash
# 1. 检查rocminfo是否可用
which rocminfo
rocminfo --version

# 2. 如果rocminfo存在，查看完整输出
rocminfo | head -20

# 3. 检查ROCm安装
ls /opt/rocm*/
dpkg -l | grep rocm

# 4. 查看GPU设备
ls /dev/kfd
ls /dev/dri/
```

#### 可能的原因和解决方案

**A. ROCm未正确安装**
```bash
# 检查是否安装
dpkg -l | grep -i rocm

# 如果没有安装，参考您的安装文档
cd ../rocm_install
cat GPU_DRIVER_INSTALL_GUIDE.md
```

**B. rocminfo命令格式变化**
```bash
# 尝试不同的命令格式
rocminfo
/opt/rocm/bin/rocminfo
rocm-smi --showproductname
```

**C. 修复脚本中的版本检测**

编辑`setup_env.sh`，将ROCm检测部分改为：
```bash
# 尝试多种方式检测ROCm
if command -v rocminfo &> /dev/null; then
    ROCM_VERSION=$(rocminfo 2>/dev/null | grep -i "rocm version" | head -1 | awk '{print $NF}')
    
    # 如果第一种方法失败，尝试其他方法
    if [ -z "$ROCM_VERSION" ]; then
        ROCM_VERSION=$(rocm-smi --showproductname 2>/dev/null | grep -i "rocm version" | awk '{print $NF}')
    fi
    
    # 如果还是失败，尝试从路径检测
    if [ -z "$ROCM_VERSION" ]; then
        ROCM_VERSION=$(ls -d /opt/rocm-* 2>/dev/null | head -1 | sed 's/.*rocm-//')
    fi
fi
```

### 问题3: GPU不可用

这是最关键的问题。PyTorch检测不到GPU有几个可能的原因：

#### 诊断步骤

```bash
# 1. 检查GPU硬件
lspci | grep -i vga
lspci | grep -i amd

# 2. 检查内核模块
lsmod | grep amdgpu
lsmod | grep kfd

# 3. 检查设备节点
ls -la /dev/kfd
ls -la /dev/dri/

# 4. 检查用户权限
groups
# 应该包含 render 和/或 video 组
```

#### 解决方案

**A. 添加用户到正确的组**
```bash
# 添加到render和video组
sudo usermod -a -G render $USER
sudo usermod -a -G video $USER

# 重新登录或重启系统使更改生效
```

**B. 检查内核模块**
```bash
# 如果amdgpu模块未加载
sudo modprobe amdgpu

# 检查是否加载
lsmod | grep amdgpu
```

**C. 重新安装AMDGPU驱动**
```bash
cd ../rocm_install
# 参考您的安装文档重新安装驱动
```

**D. 验证ROCm环境**
```bash
# 设置环境变量
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # 根据您的GPU调整
export ROCM_PATH=/opt/rocm

# 测试
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## 🚀 完整的推荐流程

### 步骤1: 修复PATH
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
uv --version  # 应该显示版本号
```

### 步骤2: 检查ROCm安装
```bash
# 检查ROCm
rocminfo
rocm-smi

# 如果没有输出或报错，需要先安装ROCm
cd ../rocm_install
cat GPU_DRIVER_INSTALL_GUIDE.md
```

### 步骤3: 配置用户权限
```bash
sudo usermod -a -G render,video $USER
# 然后重新登录
```

### 步骤4: 设置环境变量（添加到~/.bashrc）
```bash
cat >> ~/.bashrc << 'EOF'
# ROCm环境变量
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # RT9700对应的版本
EOF

source ~/.bashrc
```

### 步骤5: 重新运行setup脚本
```bash
cd gpt_train
./setup_env.sh
```

### 步骤6: 手动验证GPU
```bash
source .venv/bin/activate

python3 << 'EOF'
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"设备数量: {torch.cuda.device_count()}")
    print(f"设备名称: {torch.cuda.get_device_name(0)}")
    print(f"设备架构: {torch.cuda.get_device_capability(0)}")
else:
    print("GPU不可用，可能的原因：")
    print("1. ROCm驱动未正确安装")
    print("2. 用户权限不足")
    print("3. 环境变量未设置")
    print("4. PyTorch版本与ROCm版本不匹配")
EOF
```

---

## 📝 快速参考

### 环境变量（添加到~/.bashrc）
```bash
export PATH="$HOME/.local/bin:$PATH"
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

### 常用命令
```bash
# uv相关
uv --version
uv venv
uv pip install <package>
uv run python script.py

# ROCm相关
rocminfo
rocm-smi
/opt/rocm/bin/rocminfo

# PyTorch测试
python3 -c "import torch; print(torch.cuda.is_available())"
```

### 用户组
```bash
# 添加到必要的组
sudo usermod -a -G render,video $USER

# 查看当前组
groups

# 重新登录使更改生效
```

---

## ⚡ 如果仍然无法解决

1. **提供详细信息**：
   ```bash
   # 收集诊断信息
   echo "=== Python版本 ==="
   python3 --version
   
   echo "=== ROCm信息 ==="
   rocminfo 2>&1 | head -30
   rocm-smi
   
   echo "=== GPU设备 ==="
   lspci | grep -i amd
   ls -la /dev/kfd /dev/dri/
   
   echo "=== 内核模块 ==="
   lsmod | grep amdgpu
   
   echo "=== 用户组 ==="
   groups
   
   echo "=== PyTorch ==="
   source .venv/bin/activate
   python3 -c "import torch; print(torch.__version__)"
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

2. **查看相关文档**：
   - `../rocm_install/GPU_DRIVER_INSTALL_GUIDE.md` - ROCm安装指南
   - `../rccl_install/` - RCCL和GPU通信相关

3. **常见问题检查清单**：
   - [ ] ROCm是否正确安装？
   - [ ] 用户是否在render/video组？
   - [ ] /dev/kfd是否存在且有权限？
   - [ ] 环境变量是否设置？
   - [ ] PyTorch版本是否与ROCm兼容？
