# GPU 训练优化指南

## 问题诊断

观察到的 GPU 状态：
```
Device  GPU%  VRAM%  Power   SCLK     MCLK     Temp
0       99%   19%    292.0W  2934Mhz  1258Mhz  66.0°C
```

**分析**：
- ✅ **GPU 利用率 99%**: GPU 计算单元满负荷运行
- ❌ **VRAM 使用率 19%**: 显存严重未充分利用
- ⚠️ **功耗 292W/300W**: 接近最大功耗

**结论**: 这是典型的 **计算密集型 (Compute-Bound)** 场景，而非 **显存密集型 (Memory-Bound)**。

## 根本原因

当前配置下，GPU 能够非常快速地完成计算，但每批次处理的数据量太小，导致：

1. **批次太小** → GPU 计算单元未充分利用
2. **数据加载开销** → 等待数据的时间相对较长
3. **内核启动开销** → 频繁的小批次导致启动开销占比高

## 优化策略

### 优化目标

在不超出显存（VRAM）限制的前提下，**最大化每次迭代处理的数据量**，从而：
- 提高 GPU 并行计算效率
- 减少数据加载和内核启动的相对开销
- 缩短总训练时间

### 优化方案对比

| 优化方法 | 原始值 | 优化值 | VRAM 影响 | 训练速度 | 说明 |
|---------|--------|--------|-----------|----------|------|
| **批次大小** | 8 | 32 | ↑↑↑ | ↑↑↑ | 最直接有效 |
| **梯度累积** | 1 | 4 | → | ↑↑ | 模拟更大批次 |
| **混合精度 (FP16)** | 关闭 | 启用 | ↓↓ | ↑↑ | 减半显存 + 加速 |
| **梯度检查点** | 关闭 | 可选 | ↓↓ | ↓ | 省显存但稍慢 |
| **数据加载器** | 0 | 4 | → | ↑ | 并行加载数据 |

## 已实施的优化

### 1. 增大批次大小 (Batch Size)

**变更**: `8` → `32` (4倍)

**原理**：
```
批次大小越大 → 每次前向/反向传播处理更多样本 → GPU 并行度更高
```

**预期效果**：
- VRAM 使用率: 19% → ~60-70%
- 训练速度: 提升 2-3 倍
- 训练稳定性: 更稳定（大批次梯度估计更准确）

**显存占用估算**：
```
单样本显存 ≈ 模型参数 + 梯度 + 优化器状态 + 激活值
批次显存 ≈ 单样本显存 × 批次大小

small 模型 (117M 参数):
- FP32: 117M × 4 bytes × 4 (参数+梯度+优化器) ≈ 1.8GB
- 批次数据: 32 × 512 tokens × 4 bytes ≈ 0.06GB
- 激活值: ~0.5-1GB (取决于序列长度)
总计: ~3-4GB (批次32) vs ~1GB (批次8)
```

### 2. 梯度累积 (Gradient Accumulation)

**变更**: `1` → `4`

**原理**：
```
累积 N 个小批次的梯度 → 一次更新参数
有效批次 = 物理批次 × 累积步数
```

**优势**：
- 模拟更大的批次大小（32 × 4 = **128 有效批次**）
- 不额外占用显存
- 训练效果等同于批次128，但显存占用只相当于批次32

**示例**：
```python
# 传统方式 (批次128, 可能 OOM)
for batch in dataloader:  # batch_size=128
    loss = model(batch)
    loss.backward()
    optimizer.step()

# 梯度累积 (等效批次128, 显存占用相当于32)
for i, batch in enumerate(dataloader):  # batch_size=32
    loss = model(batch)
    loss = loss / 4  # 缩放损失
    loss.backward()  # 累积梯度
    
    if (i + 1) % 4 == 0:
        optimizer.step()  # 每4步更新一次
        optimizer.zero_grad()
```

### 3. 混合精度训练 (FP16) - 可选

**启用方式**: 添加 `--fp16` 参数

**原理**：
```
FP32 (32位浮点) → FP16 (16位浮点)
显存占用减半，计算速度提升
```

**优势**：
- **显存减半**: 模型参数、梯度从 4 bytes → 2 bytes
- **计算加速**: RDNA 架构对 FP16 有硬件加速
- **功耗降低**: 可能降低 10-20% 功耗

**注意事项**：
- ROCm 对 FP16 支持良好，但可能有轻微精度损失
- 使用损失缩放 (Loss Scaling) 避免数值下溢
- 对大多数训练任务影响可忽略

**建议**：
```bash
# 如果 VRAM 仍然不足，启用 FP16
ENABLE_FP16="--fp16"
```

### 4. 数据加载优化

**变更**: `num_workers=0` → `num_workers=4`

**原理**：
```
多进程并行加载数据 → 减少 GPU 等待时间
```

**效果**：
- CPU 多核并行预处理数据
- GPU 计算和数据加载重叠
- 减少 I/O 瓶颈

### 5. 梯度检查点 (Gradient Checkpointing) - 可选

**启用方式**: 添加 `--gradient_checkpointing` 参数

**原理**：
```
训练时不保存所有中间激活值
反向传播时重新计算 → 显存换时间
```

**适用场景**：
- 显存不足时
- 模型层数很多时
- 可接受 10-20% 的训练速度降低

## 使用指南

### 快速开始

1. **赋予执行权限**：
```bash
chmod +x run_single_gpu_optimized.sh
```

2. **运行优化训练**：
```bash
./run_single_gpu_optimized.sh
```

3. **监控 GPU 状态**：
```bash
# 另一个终端
watch -n 1 rocm-smi
```

### 参数调优流程

#### 步骤 1: 基线测试

使用默认参数运行，观察 VRAM 使用率：

```bash
# 默认: batch_size=32, gradient_accumulation=4
./run_single_gpu_optimized.sh
```

#### 步骤 2: 根据 VRAM 使用率调整

**场景 A: VRAM 使用率 < 60%**

继续增大批次：

```bash
# 编辑 run_single_gpu_optimized.sh
BATCH_SIZE=48  # 或 64
```

**场景 B: VRAM 使用率 70-80%** ✅

这是理想状态，保持当前配置。

**场景 C: 遇到 OOM (Out of Memory)**

减小批次或启用优化：

```bash
# 选项1: 减小批次
BATCH_SIZE=24  # 或 16

# 选项2: 启用混合精度
ENABLE_FP16="--fp16"

# 选项3: 启用梯度检查点
ENABLE_GRAD_CKPT="--gradient_checkpointing"
```

#### 步骤 3: 优化有效批次大小

如果需要保持大的有效批次（提高训练稳定性）：

```bash
# 减小物理批次，增大梯度累积
BATCH_SIZE=16
GRADIENT_ACCUM=8
# 有效批次 = 16 × 8 = 128 (与之前相同)
```

### 不同模型大小的建议配置

#### Tiny 模型 (50M 参数)

```bash
MODEL_SIZE="tiny"
BATCH_SIZE=64        # 可以很大
GRADIENT_ACCUM=2
ENABLE_FP16=""       # 通常不需要
```

**预期 VRAM**: ~2-3GB

#### Small 模型 (117M 参数) - 默认

```bash
MODEL_SIZE="small"
BATCH_SIZE=32
GRADIENT_ACCUM=4
ENABLE_FP16=""       # 可选
```

**预期 VRAM**: ~4-6GB

#### Medium 模型 (345M 参数)

```bash
MODEL_SIZE="medium"
BATCH_SIZE=16        # 较小
GRADIENT_ACCUM=8     # 较大
ENABLE_FP16="--fp16" # 建议启用
```

**预期 VRAM**: ~8-12GB

## 性能对比

### 预期提升

基于您的配置 (RT9700, ~16GB VRAM):

| 配置 | 批次大小 | 有效批次 | VRAM 使用 | 相对速度 | 训练时间 |
|------|---------|---------|----------|----------|---------|
| **原始** | 8 | 8 | ~19% | 1.0x | 基线 |
| **优化 (默认)** | 32 | 128 | ~60% | **3-4x** | **25-33%** |
| **优化 + FP16** | 32 | 128 | ~40% | **4-5x** | **20-25%** |
| **激进优化** | 48 | 192 | ~80% | **5-6x** | **16-20%** |

### 实际测试建议

运行对比测试：

```bash
# 1. 原始配置
./run_single_gpu.sh

# 2. 优化配置
./run_single_gpu_optimized.sh

# 比较:
# - 每步训练时间 (samples/sec)
# - VRAM 使用率
# - GPU 利用率
# - 总训练时间
```

## 监控和诊断

### 实时监控

```bash
# 终端 1: 运行训练
./run_single_gpu_optimized.sh

# 终端 2: 监控 GPU
watch -n 1 'rocm-smi | grep -A 1 "Device"'

# 终端 3: 监控进程
htop -p $(pgrep -f train_single_gpu_optimized)
```

### 关键指标

观察以下指标判断优化效果：

1. **VRAM 使用率**: 目标 60-80%
   ```
   - < 50%: 可以增大批次
   - 60-80%: 理想状态 ✅
   - > 90%: 可能需要减小批次
   ```

2. **GPU 利用率**: 应保持 90%+
   ```
   - 如果下降，可能是数据加载瓶颈
   - 增大 num_workers
   ```

3. **训练速度** (samples/sec):
   ```
   - 原始: ~100 samples/sec
   - 优化: ~300-400 samples/sec
   ```

4. **功耗**:
   ```
   - FP32: ~280-300W
   - FP16: ~250-280W
   ```

### 常见问题

#### Q1: 遇到 OOM 错误

```bash
RuntimeError: HIP out of memory
```

**解决方案**：
```bash
# 1. 减小批次
BATCH_SIZE=16

# 2. 启用混合精度
ENABLE_FP16="--fp16"

# 3. 启用梯度检查点
ENABLE_GRAD_CKPT="--gradient_checkpointing"

# 4. 减小序列长度
MAX_LENGTH=256
```

#### Q2: 训练速度没有提升

可能原因：
- 数据加载瓶颈 → 增大 `num_workers`
- 磁盘 I/O 瓶颈 → 使用 SSD 或缓存数据集
- CPU 瓶颈 → 检查 CPU 使用率

#### Q3: 训练不稳定/损失震荡

大批次可能需要调整学习率：

```bash
# 线性缩放规则
原始: batch_size=8, lr=5e-5
优化: batch_size=32, lr=2e-4  # 4倍批次 → 4倍学习率
```

或使用学习率预热：

```python
# 在 train_single_gpu_optimized.py 中
training_args = TrainingArguments(
    ...
    warmup_steps=500,  # 添加预热
)
```

## 高级优化

### 1. 动态批次大小

根据序列长度动态调整批次：

```python
# 短序列 → 大批次
# 长序列 → 小批次
```

### 2. 混合精度 + TF32

ROCm 支持 TF32（如果硬件支持）：

```python
torch.set_float32_matmul_precision('high')  # 使用 TF32
```

### 3. 编译优化

PyTorch 2.0+ 支持编译优化：

```python
model = torch.compile(model)  # 可能提速 10-30%
```

## 总结

### 核心原则

1. **最大化 VRAM 利用率**: 目标 60-80%
2. **平衡批次大小和梯度累积**: 有效批次要足够大
3. **启用硬件加速**: FP16, TF32
4. **优化数据流水线**: 多进程加载，预取

### 推荐起始配置

```bash
MODEL_SIZE="small"
BATCH_SIZE=32
GRADIENT_ACCUM=4
ENABLE_FP16=""           # 如果 VRAM 不够再启用
NUM_WORKERS=4
```

### 下一步

1. 运行优化版本，观察 VRAM 使用率
2. 如果 VRAM < 60%，增大 `BATCH_SIZE`
3. 如果遇到 OOM，启用 `--fp16`
4. 监控训练指标，确保模型收敛
5. 根据实际情况微调参数

**预期结果**: 训练速度提升 **3-5倍**，VRAM 使用率达到 **60-80%** ✅
