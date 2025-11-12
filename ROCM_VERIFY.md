# ROCm PyTorch GPU 验证指南

本指南说明如何在 ROCm Docker 容器中验证 PyTorch GPU 功能。

## 快速验证

### 方法 1: 使用验证脚本（推荐）

在 Docker 容器中运行完整的验证脚本：

```bash
# 进入 Docker 容器
cd /home/xilinx/Documents/min/gpt_train
./docker_run.sh

# 在容器内运行验证脚本
python3 test_rocm_pytorch.py
```

该脚本会自动检测：
- ✓ PyTorch 安装状态
- ✓ ROCm/HIP 版本
- ✓ GPU 可用性和数量
- ✓ GPU 设备信息（型号、显存等）
- ✓ 基本 GPU 操作（张量创建、运算、数据传输等）
- ✓ GPU 性能测试
- ✓ ROCm SMI 系统信息

### 方法 2: 简单命令验证

最简单的验证方式：

```bash
# 在 Docker 容器内
python3 -c "import torch; print('PyTorch:', torch.__version__); print('GPU 可用:', torch.cuda.is_available()); print('GPU 数量:', torch.cuda.device_count()); print('GPU 设备:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### 方法 3: 交互式验证

```bash
# 在 Docker 容器内启动 Python
python3
```

然后逐步测试：

```python
import torch

# 检查版本
print("PyTorch 版本:", torch.__version__)
print("HIP 版本:", torch.version.hip if hasattr(torch.version, 'hip') else 'N/A')

# 检查 GPU
print("GPU 可用:", torch.cuda.is_available())
print("GPU 数量:", torch.cuda.device_count())

# 如果 GPU 可用
if torch.cuda.is_available():
    print("GPU 设备:", torch.cuda.get_device_name(0))
    
    # 测试 GPU 运算
    x = torch.randn(100, 100, device='cuda')
    y = torch.randn(100, 100, device='cuda')
    z = torch.matmul(x, y)
    print("GPU 运算测试: ✓ 成功")
```

## 常见问题排查

### 问题 1: GPU 不可用

如果 `torch.cuda.is_available()` 返回 `False`：

1. **检查 Docker 容器配置**
   ```bash
   # 确保 Docker 启动时包含以下参数
   --device=/dev/kfd --device=/dev/dri
   ```

2. **检查环境变量**
   ```bash
   echo $HSA_OVERRIDE_GFX_VERSION
   echo $PYTORCH_ROCM_ARCH
   ```

3. **运行修复脚本**
   ```bash
   source fix_hip_env.sh
   ```

### 问题 2: HIP 错误

如果遇到 HIP 相关错误：

```bash
# 在容器内运行
source fix_hip_env.sh
python3 test_rocm_pytorch.py
```

### 问题 3: 显存不足

```python
import torch
# 清理 GPU 缓存
torch.cuda.empty_cache()

# 查看显存使用
print(f"已分配: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"已缓存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
```

## 验证 ROCm 系统信息

### 查看 GPU 状态

```bash
# 在容器内
rocm-smi
```

### 查看 GPU 架构

```bash
rocminfo | grep "Name:" | grep "gfx"
```

### 查看 ROCm 版本

```bash
apt list --installed | grep rocm
```

## 性能测试

验证脚本包含性能测试，会比较 CPU 和 GPU 的矩阵运算速度。

期望结果：
- GPU 应该比 CPU 快 5-50 倍（取决于具体硬件）
- 如果 GPU 比 CPU 慢，说明配置可能有问题

## Docker 容器启动检查清单

确保 Docker 容器正确配置：

- [x] 设备挂载: `--device=/dev/kfd --device=/dev/dri`
- [x] 用户组: `--group-add video --group-add render`
- [x] 共享内存: `--shm-size 8G`（或更大）
- [x] 环境变量: `HSA_OVERRIDE_GFX_VERSION` 和 `PYTORCH_ROCM_ARCH`

## 下一步

验证成功后，可以：

1. **运行训练脚本**
   ```bash
   python3 train_single_gpu.py --model_size tiny
   ```

2. **测试文本生成**
   ```bash
   python3 test_generation.py
   ```

3. **多 GPU 训练**
   ```bash
   python3 train_multi_gpu.py --model_size small
   ```

## 相关文档

- [Docker 设置指南](DOCKER_SETUP.md)
- [HIP 错误修复](FIX_HIP_ERROR.md)
- [快速修复指南](QUICK_FIX.md)
- [主要 README](README.md)
