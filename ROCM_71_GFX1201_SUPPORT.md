# ROCm 7.1 对 gfx1201 的支持情况

根据 [AMD ROCm 官方兼容性文档](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html#rdna-os-700)：

**ROCm 7.0.x 和 7.1.0 已经正式支持 gfx1201 (AMD Radeon PRO AI PRO R9700)！**

## 📋 支持详情

### ROCm 版本
- ✅ **ROCm 7.1.0** - 支持 gfx1201
- ✅ **ROCm 7.0.2** - 支持 gfx1201

### PyTorch 版本
- ROCm 7.1.0: **PyTorch 2.8, 2.7, 2.6**
- ROCm 7.0.2: **PyTorch 2.8, 2.7, 2.6**

### 支持的操作系统（重要！）

**对于 gfx1201，仅支持以下操作系统**：
- Ubuntu 24.04.3
- Ubuntu 22.04.5
- RHEL 10.0
- RHEL 9.6

注意：gfx1201 **不支持** RHEL 9.4, RHEL 8.10, SLES 等其他操作系统。

## 🔍 为什么你的环境还不工作？

你当前使用的镜像：
```
rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.8.0
```

**可能的问题**：

1. **PyTorch 编译目标**
   - 虽然 ROCm 7.1 支持 gfx1201
   - 但 PyTorch 镜像可能没有为 gfx1201 编译
   - 需要确认 PyTorch 是否包含 gfx1201 的代码对象

2. **Ubuntu 版本**
   - 你的镜像是 Ubuntu 22.04
   - 需要确认是否是 22.04.5 版本

## ✅ 解决方案

### 方案 1: 验证当前镜像是否包含 gfx1201 支持

在 Docker 容器中运行：

```bash
# 检查 PyTorch 支持的架构
python3 -c "import torch; print(torch.cuda.get_arch_list())"

# 检查 HIP/ROCm 库
ls -la /opt/rocm/lib/ | grep gfx12

# 检查 Ubuntu 版本
cat /etc/os-release | grep VERSION
```

如果输出中包含 `gfx1201` 或 `gfx12`，说明镜像已支持。

### 方案 2: 使用最新的 ROCm 7.1 镜像

```bash
# 拉取最新镜像
docker pull rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.8.0

# 或者尝试 latest 标签
docker pull rocm/pytorch:latest
```

### 方案 3: 使用 Ubuntu 24.04 的镜像

根据官方文档，gfx1201 在 Ubuntu 24.04.3 上有更好的支持：

```bash
# 检查是否有 Ubuntu 24.04 的镜像
docker search rocm/pytorch | grep 24.04

# 如果有，拉取
docker pull rocm/pytorch:rocm7.1_ubuntu24.04_py3.10_pytorch_release_2.8.0
```

### 方案 4: 自己编译 PyTorch for gfx1201

如果官方镜像确实不包含 gfx1201 支持：

```bash
# 在容器内
git clone https://github.com/pytorch/pytorch.git
cd pytorch

# 设置编译目标
export PYTORCH_ROCM_ARCH=gfx1201

# 编译（需要几小时）
python3 tools/amd_build/build_amd.py
python3 setup.py install
```

## 🎯 推荐的测试步骤

### 步骤 1: 检查镜像支持

```bash
# 在 Docker 容器内
cd /workspace

# 创建测试脚本
cat > check_gfx1201.py << 'EOF'
import torch
import sys

print("PyTorch 版本:", torch.__version__)
print("ROCm 版本:", torch.version.hip if hasattr(torch.version, 'hip') else 'N/A')

# 检查支持的架构
try:
    archs = torch.cuda.get_arch_list()
    print("\n支持的架构列表:")
    for arch in archs:
        print(f"  - {arch}")
    
    if 'gfx1201' in str(archs) or 'gfx12' in str(archs):
        print("\n✓ 镜像支持 gfx1201!")
    else:
        print("\n✗ 镜像不支持 gfx1201")
        print("   需要使用兼容模式 (gfx1101)")
except Exception as e:
    print(f"错误: {e}")

# 检查 GPU
print(f"\nGPU 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
EOF

python3 check_gfx1201.py
```

### 步骤 2: 根据结果选择策略

#### 如果镜像支持 gfx1201

```bash
# 使用原生配置
export HSA_OVERRIDE_GFX_VERSION=12.0.1
export PYTORCH_ROCM_ARCH=gfx1201
export AMD_SERIALIZE_KERNEL=1
export GPU_MAX_HW_QUEUES=4
```

#### 如果镜像不支持 gfx1201

```bash
# 使用兼容模式（你当前的配置）
export HSA_OVERRIDE_GFX_VERSION=11.0.1
export PYTORCH_ROCM_ARCH=gfx1101
export AMD_SERIALIZE_KERNEL=3
export GPU_MAX_HW_QUEUES=1
```

## 📝 更新 docker_run.sh 的建议

如果验证后发现镜像支持 gfx1201，可以更新配置：

```bash
# 编辑 docker_run.sh，修改环境变量为：
  -e HSA_OVERRIDE_GFX_VERSION=12.0.1 \
  -e PYTORCH_ROCM_ARCH=gfx1201 \
  -e AMD_SERIALIZE_KERNEL=1 \
  -e GPU_MAX_HW_QUEUES=2 \
  -e HSA_ENABLE_SDMA=0 \
  -e PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:256 \
```

## 🔧 诊断命令合集

```bash
# 在 Docker 容器内运行

# 1. 检查 Ubuntu 版本
cat /etc/os-release

# 2. 检查 ROCm 版本
cat /opt/rocm/.info/version

# 3. 检查 PyTorch 编译信息
python3 -c "import torch; print(torch.__config__.show())"

# 4. 检查 HIP 库中的 gfx 支持
find /opt/rocm -name "*gfx12*" 2>/dev/null | head -10

# 5. 检查实际 GPU
rocminfo | grep -i "name.*gfx"
```

## 🎯 最终建议

### 立即行动
1. **先验证**当前镜像是否已包含 gfx1201 支持
2. **如果支持**，更新为原生 gfx1201 配置
3. **如果不支持**，继续使用 gfx1101 兼容模式

### 长期方案
等待 AMD/PyTorch 发布包含 gfx1201 编译目标的更新镜像，或者考虑从源码编译。

## 📚 参考

- [AMD ROCm 兼容性矩阵](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html#rdna-os-700)
- [ROCm 支持的 GPU](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus)
- [PyTorch ROCm 文档](https://pytorch.org/docs/stable/notes/hip.html)
