# 双节点GPT训练快速指南（1Gbps网络）

## 🎯 目标

在两个RX9700节点上训练GPT-2 Tiny模型，使用1Gbps网络连接。

## ✅ 可行性结论

**1Gbps网络完全可行！**

- 网络传输量：每次梯度同步 ~201MB（FP16）
- 推荐配置下通信占比：~20%
- 训练效率：~80%（相比理想情况）
- 加速比：约1.6倍（相比单节点）

详细计算见：`NETWORK_BANDWIDTH_ANALYSIS.md`

## 🚀 快速开始（3步）

### 前置要求

1. ✅ 两个节点都安装ROCm和PyTorch
2. ✅ 网络互通（能ping通）
3. ✅ SSH免密登录
4. ✅ 共享存储或代码同步

### 步骤1：在两个节点上准备代码

```bash
# 两个节点都执行
cd /path/to/shared/gpt_train
# 或确保代码在两个节点的相同路径
```

### 步骤2：启动训练

#### 主节点（node1，假设IP: 192.168.1.100）

```bash
./run_2nodes.sh 0 192.168.1.100
```

#### 从节点（node2，假设IP: 192.168.1.101）

```bash
./run_2nodes.sh 1 192.168.1.100
```

### 步骤3：监控训练

```bash
# 在主节点，另一个终端
watch -n 1 rocm-smi

# 查看训练日志
tensorboard --logdir=./output_2nodes_tiny/logs
```

## 📊 推荐配置详解

### 默认配置（已优化）

```bash
模型: GPT-2 Tiny (50M参数)
节点数: 2
GPU/节点: 1
批次大小/GPU: 16
梯度累积: 8步 ⭐关键
混合精度: FP16 ✅
序列长度: 512

有效批次大小 = 16 × 2 × 8 = 256
```

### 为什么这样配置？

#### 1. 梯度累积8步
- **目的**：减少通信频率
- **效果**：每8步才同步一次梯度
- **通信占比**：从67%降到20%

#### 2. FP16混合精度
- **目的**：减少传输量
- **效果**：梯度大小减半（201MB vs 402MB）
- **额外好处**：加速训练，节省显存

#### 3. 批次大小16
- **目的**：充分利用GPU
- **平衡**：计算和通信的比例

## 📈 训练性能预测

### 以5轮训练为例

假设：
- 数据集：中文维基10%（约100k样本）
- 序列长度：512
- 有效批次：256

```
总步数 ≈ 100,000 / 256 × 5 = 1,953步
梯度同步次数 = 1,953 / 8 = 244次

网络传输总量 = 244 × 201MB = 48.8 GB
网络传输时间 = 48.8GB / 0.1GB/s = 488秒 ≈ 8分钟

GPU计算时间（估算） ≈ 1,953 × 1秒 = 33分钟
总训练时间 ≈ 41分钟

通信占比 = 8 / 41 = 19.5% ✅
```

### 与单节点对比

| 指标 | 单节点 | 双节点 | 提升 |
|------|--------|--------|------|
| 有效批次 | 128 | 256 | 2x |
| 训练时间 | ~65分钟 | ~41分钟 | 1.6x |
| GPU利用率 | 100% | 80% | - |

## 🔧 不同模型的配置

### Tiny（推荐起步）

```bash
# 主节点
./run_2nodes.sh 0 192.168.1.100 tiny 5

# 从节点
./run_2nodes.sh 1 192.168.1.100 tiny 5
```

### Small（需要调整）

```bash
# Small模型（117M参数）通信量更大
# 建议增加梯度累积到16步

# 主节点
./run_2nodes.sh 0 192.168.1.100 small 3

# 注意：run_2nodes.sh会自动使用合适的配置
```

## 🌐 网络环境变量

脚本已自动设置，但您可以根据实际情况调整：

```bash
# 网络接口名称（根据实际调整）
export NCCL_SOCKET_IFNAME=eth0  # 或 eno1, enp0s1等

# 如果没有InfiniBand
export NCCL_IB_DISABLE=1

# 调试级别（训练稳定后可以关闭）
export NCCL_DEBUG=INFO  # 或 WARN

# RCCL缓冲区大小
export NCCL_BUFFSIZE=2097152
```

查看网络接口名称：
```bash
ip addr show
# 或
ifconfig
```

## 🔍 故障排查

### 1. 节点无法通信

```bash
# 检查网络连通性
ping 192.168.1.101

# 检查端口
telnet 192.168.1.100 29500

# 检查防火墙
sudo ufw status
sudo ufw allow 29500
```

### 2. 训练速度很慢

```bash
# 查看NCCL日志，检查是否使用了正确的网络接口
# 日志中应该看到类似：
# [Rank 0] NET/Socket : Using interface eth0
```

### 3. SSH密钥问题

```bash
# 配置免密登录
ssh-keygen -t rsa
ssh-copy-id user@node2

# 测试
ssh node2 "hostname"
```

## 📝 完整流程

### 准备阶段（一次性）

```bash
# 两个节点都执行

# 1. 安装环境
cd gpt_train
./setup_env.sh

# 2. 配置SSH免密（如果需要）
ssh-keygen -t rsa
ssh-copy-id user@other_node

# 3. 同步代码（或使用NFS共享）
# 确保两个节点的代码在相同路径
```

### 训练阶段

```bash
# 主节点（192.168.1.100）
cd gpt_train
./run_2nodes.sh 0 192.168.1.100 tiny 5

# 从节点（192.168.1.101），几乎同时运行
cd gpt_train  
./run_2nodes.sh 1 192.168.1.100 tiny 5
```

### 监控阶段

```bash
# 在主节点
# 终端1：查看训练进度（会自动显示）
# 终端2：监控GPU
watch -n 1 rocm-smi

# 终端3：查看TensorBoard
tensorboard --logdir=./output_2nodes_tiny/logs
```

## 💡 优化建议

### 如果网络仍然是瓶颈

1. **增加梯度累积**
   ```bash
   # 修改run_2nodes.sh中的GRAD_ACCUM
   GRAD_ACCUM=16  # 从8增加到16
   ```

2. **减少logging频率**
   - 减少日志输出可能略微提升性能

3. **升级网络**
   - 1Gbps → 10Gbps：效率从80%提升到98%
   - 强烈推荐用于larger模型

## 🎯 总结

### ✅ 推荐配置（已内置）

```bash
模型: Tiny (50M)
网络: 1Gbps
FP16: 启用 ✅
梯度累积: 8步 ✅
批次/GPU: 16

预期效果:
- 通信占比: 20%
- 训练效率: 80%
- 加速比: 1.6x
```

### 🚀 立即使用

```bash
# 主节点
./run_2nodes.sh 0 <master_ip>

# 从节点
./run_2nodes.sh 1 <master_ip>
```

就这么简单！脚本已经配置好所有优化参数。
