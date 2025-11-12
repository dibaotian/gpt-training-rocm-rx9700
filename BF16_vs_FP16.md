# BF16 vs FP16：选择混合精度训练格式

## 🎯 什么是BF16？

**BFloat16 (BF16)** 是Google开发的16位浮点格式：
- 与FP16一样是16位
- 但有不同的位分配
- 更接近FP32的数值范围

## 📊 格式对比

| 格式 | 总位数 | 指数位 | 尾数位 | 数值范围 |
|------|--------|--------|--------|---------|
| FP32 | 32 | 8 | 23 | ±3.4×10³⁸ |
| BF16 | 16 | 8 | 7 | ±3.4×10³⁸ |
| FP16 | 16 | 5 | 10 | ±6.5×10⁴ |

**关键差异**：
- **BF16**：指数位=8（与FP32相同），数值范围大
- **FP16**：指数位=5，数值范围小，容易溢出

## ✅ BF16的优势

### 1. 数值稳定性 ⭐⭐⭐⭐⭐

```python
# BF16不容易出现：
- 梯度下溢（gradient underflow）
- 梯度爆炸（gradient explosion）
- Loss变成NaN
```

### 2. 与FP32转换简单

```
BF16 ↔ FP32: 只需截断/扩展尾数
FP16 ↔ FP32: 需要重新计算指数和尾数
```

### 3. 训练更稳定

- 不需要复杂的loss scaling
- 更少的NaN/Inf问题
- 更适合大模型训练

### 4. 显存节省相同

```
BF16显存 = FP16显存 = FP32显存 / 2
```

## ⚖️ BF16 vs FP16 对比

| 特性 | BF16 | FP16 |
|------|------|------|
| 显存节省 | 50% ✅ | 50% ✅ |
| 数值稳定性 | 高 ✅✅✅ | 中 ⚠️ |
| 数值范围 | 大（同FP32） | 小 |
| 精度 | 较低 | 较高 |
| 训练收敛 | 好 ✅✅ | 可能不稳定 |
| 硬件支持 | 较新GPU | 广泛支持 |
| ROCm支持 | ✅ ROCm 5.0+ | ✅ 所有版本 |

## 🎯 何时使用BF16？

### ✅ 推荐使用BF16

1. **大模型训练**（Large, XL）
2. **训练不稳定**（FP16出现NaN）
3. **需要更好收敛**
4. **ROCm 5.0+**（支持BF16）

### ✅ 可以使用FP16

1. **小模型训练**（Tiny, Small）
2. **推理任务**
3. **需要更高精度**
4. **旧版本ROCm**（不支持BF16）

## 🚀 使用方法

### 单GPU训练

```bash
# 使用BF16
python3 train_single_gpu.py \
    --model_size small \
    --use_chinese \
    --bf16 \
    --batch_size 16

# 使用FP16
python3 train_single_gpu.py \
    --model_size small \
    --use_chinese \
    --fp16 \
    --batch_size 16
```

### 双节点训练

```bash
# BF16（默认，推荐）
./run_2nodes.sh 0 192.168.1.100 tiny 5

# 或手动指定FP16
torchrun ... train_multi_gpu.py --fp16
```

## 📊 实际表现对比

### 训练稳定性

```python
# FP16可能出现
Loss: 2.456 → 1.834 → NaN → 训练失败 ❌

# BF16更稳定
Loss: 2.456 → 1.834 → 1.523 → 1.245 → 收敛 ✅
```

### 网络传输量（完全相同）

```
GPT-2 Tiny:
- BF16梯度: 101 MB
- FP16梯度: 101 MB
- All-Reduce: 201 MB（两者相同）
```

### 训练速度

```
GPU计算速度：BF16 ≈ FP16（几乎相同）
网络传输：完全相同
```

## 💡 推荐策略

### 默认选择：BF16 ✅

```bash
# 单GPU
./train_chinese.sh small 5  # 自动使用最佳配置

# 双节点
./run_2nodes.sh 0 192.168.1.100  # 默认BF16
```

### 切换到FP16

如果BF16不支持或想尝试FP16：

```bash
# 修改脚本，将--bf16改为--fp16
# 或手动运行
python3 train_single_gpu.py --fp16 --model_size small
```

## 🔍 检查ROCm BF16支持

```bash
python3 << 'EOF'
import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"ROCm版本: {torch.version.hip}")

# 测试BF16
try:
    x = torch.randn(100, 100).bfloat16().cuda()
    y = x + x
    print("✓ BF16支持正常")
except Exception as e:
    print(f"❌ BF16不支持: {e}")
    print("建议使用FP16")
EOF
```

## 🎓 最佳实践

### 1. 大模型优先BF16

```bash
# Large/XL模型
python3 train_single_gpu.py \
    --model_size large \
    --bf16  # 更稳定
```

### 2. 监控Loss曲线

```bash
tensorboard --logdir=./output_*/logs
# 检查是否有NaN或异常波动
```

### 3. 如果遇到问题

```bash
# BF16遇到问题 → 尝试FP16
# FP16遇到NaN → 切换到BF16
# 都有问题 → 使用FP32
```

## 📝 总结

### ✅ 双节点训练推荐配置

```
模型: GPT-2 Tiny/Small
混合精度: BF16 ✅ (默认)
梯度累积: 8步
网络: 1Gbps
效率: ~80%
```

**BF16优势**：
- ✅ 与FP16相同的显存节省
- ✅ 与FP16相同的传输量
- ✅ 更好的数值稳定性
- ✅ 更少的训练问题

现在脚本已默认使用BF16，您可以直接开始训练！
