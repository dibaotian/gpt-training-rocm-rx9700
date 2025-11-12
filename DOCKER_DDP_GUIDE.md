# Docker 跨节点 DDP 训练完整指南

## 目录

- [🚀 快速开始](#-快速开始)
- [📋 架构概述](#-架构概述)
- [🔑 关键配置要点](#-关键配置要点)
- [📝 详细配置说明](#-详细配置说明)
- [🔍 故障排查](#-故障排查)
- [📊 性能优化建议](#-性能优化建议)
- [💡 最佳实践](#-最佳实践)
- [📈 预期性能](#-预期性能)

---

## 🚀 快速开始

### 前置条件检查

```bash
# 两个节点都需要有:
✓ Docker
✓ AMD GPU + ROCm驱动
✓ 网络互通
```

### 5分钟快速启动

#### 步骤1: 准备代码（两个节点）

```bash
# 确保代码在两个节点的相同路径
cd /path/to/gpt_train

# 赋予脚本执行权限
chmod +x docker_run_ddp.sh test_docker_ddp_network.sh
```

#### 步骤2: 测试网络（可选但推荐）

**在从节点上运行：**

```bash
# 替换为实际的主节点IP
./test_docker_ddp_network.sh 10.161.176.100

# 如果网络接口不是eth0，指定实际接口名
./test_docker_ddp_network.sh 10.161.176.100 ens13f3
```

查看网络接口名：
```bash
ip addr show
# 或
ifconfig
```

#### 步骤3: 启动训练

**主节点（例如 192.168.1.100）：**

```bash
# 基础用法（使用默认配置）
./docker_run_ddp.sh 0 10.161.176.100

# 或指定所有参数
./docker_run_ddp.sh 0 10.161.176.100 tiny 5 ens13f3
```

**从节点（例如 192.168.1.101）：**

```bash
# 使用相同的参数（除了rank）
./docker_run_ddp.sh 1 10.161.176.100

# 或
./docker_run_ddp.sh 1 10.161.176.100 tiny 5 ens13f3
```

#### 步骤4: 等待训练完成

**首次运行会自动安装依赖：**
```
检查Python依赖...
首次运行，正在安装依赖包...
✓ 依赖安装完成
```

然后容器会自动启动训练。你会看到类似的输出：

```
========================================
Docker DDP 容器已启动
========================================

节点信息:
  节点Rank: 0
  主节点: 192.168.1.100:29500
  World Size: 2

...

🚀 开始训练...
========================================

Epoch 1/5:   0%|          | 0/244 [00:00<?, ?it/s]
```

### 📊 参数说明

#### docker_run_ddp.sh 参数

```bash
./docker_run_ddp.sh <node_rank> <master_addr> [model_size] [epochs] [network_interface]
```

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| node_rank | 节点序号 | 必需 | 0（主节点）或 1（从节点） |
| master_addr | 主节点IP | 必需 | 192.168.1.100 |
| model_size | 模型大小 | tiny | tiny/small/medium |
| epochs | 训练轮数 | 5 | 3, 5, 10 |
| network_interface | 网络接口 | eth0 | eno1, enp0s1 |

#### 模型配置对照

| 模型 | 参数量 | 批次大小 | 梯度累积 | 有效批次 |
|------|--------|----------|----------|----------|
| tiny | 50M | 16 | 8 | 256 |
| small | 117M | 8 | 16 | 256 |
| medium | 345M | 4 | 32 | 256 |

**监控训练（可选）：**

在主节点另开终端：
```bash
# 监控GPU
watch -n 1 rocm-smi

# 或查看日志
docker logs -f gpt-train-node0
```

**等待训练完成**

训练完成后，模型会保存在：
- `./gpt_model_docker_ddp_tiny/`
- `./output_docker_ddp_tiny/`

---

## 📋 架构概述

### 系统架构

```
节点1 (192.168.1.100)          节点2 (192.168.1.101)
┌─────────────────────┐        ┌─────────────────────┐
│  Docker 容器        │        │  Docker 容器        │
│  ┌───────────────┐  │        │  ┌───────────────┐  │
│  │ Rank 0 (主)   │  │◄──────►│  │ Rank 1 (从)   │  │
│  │ GPU 0         │  │  网络  │  │ GPU 0         │  │
│  │ 端口:29500    │  │        │  │ 端口:29500    │  │
│  └───────────────┘  │        │  └───────────────┘  │
└─────────────────────┘        └─────────────────────┘
         │                              │
         └──────────────┬───────────────┘
                   共享存储 (可选)
                   或代码同步
```

### 工作原理

1. **主节点（Rank 0）**：
   - 启动 DDP 主进程
   - 监听端口 29500
   - 协调分布式训练
   - 同步模型参数

2. **从节点（Rank 1+）**：
   - 连接主节点
   - 接收训练指令
   - 同步梯度更新
   - 参与集体通信

3. **通信机制**：
   - 使用 NCCL/RCCL 后端
   - TCP/IP 网络通信
   - AllReduce 梯度同步

---

## 🔑 关键配置要点

### 1. Docker 网络模式选择

**推荐：使用 host 网络模式**

```bash
docker run --network host ...
```

**优点：**
- 容器直接使用主机网络，无需端口映射
- 性能最佳，延迟最低
- 配置简单

**缺点：**
- 容器与主机共享网络命名空间
- 端口冲突需要注意

**替代：bridge 模式 + 端口映射**
```bash
docker run -p 29500:29500 ...
```

### 2. 必须配置的环境变量

```bash
# PyTorch DDP 基础配置
MASTER_ADDR=192.168.1.100    # 主节点IP
MASTER_PORT=29500             # 通信端口
WORLD_SIZE=2                  # 总进程数
RANK=0 或 1                   # 当前进程rank

# RCCL/NCCL 网络配置
NCCL_SOCKET_IFNAME=eth0       # 网络接口名
NCCL_IB_DISABLE=1             # 禁用InfiniBand（如果没有）
NCCL_DEBUG=INFO               # 调试级别

# GPU 配置
HSA_OVERRIDE_GFX_VERSION=12.0.1
PYTORCH_ROCM_ARCH=gfx1201
```

### 3. GPU 设备映射

必须映射 GPU 设备到容器：

```bash
--device=/dev/kfd \
--device=/dev/dri \
--group-add video \
--group-add render \
```

### 4. 共享内存配置

DDP 需要大量共享内存：

```bash
--ipc=host \
--shm-size=8G \
```

---

## 📝 详细配置说明

### Docker 启动参数详解

```bash
docker run -it --rm \
  --name gpt-train-node0 \              # 容器名称
  --network host \                       # 使用主机网络
  --device=/dev/kfd \                    # AMD GPU 内核驱动
  --device=/dev/dri \                    # AMD GPU DRI
  --group-add video \                    # 视频组权限
  --group-add render \                   # 渲染组权限
  --cap-add=SYS_PTRACE \                # 调试权限
  --security-opt seccomp=unconfined \   # 安全配置
  --ipc=host \                          # 共享IPC命名空间
  --shm-size=8G \                       # 共享内存大小
  -v /path/to/code:/workspace \         # 挂载代码
  -w /workspace \                        # 工作目录
  -e MASTER_ADDR=192.168.1.100 \        # DDP主节点
  -e MASTER_PORT=29500 \                # DDP端口
  -e RANK=0 \                           # 当前rank
  -e WORLD_SIZE=2 \                     # 总进程数
  -e NCCL_SOCKET_IFNAME=eth0 \          # 网络接口
  -e NCCL_IB_DISABLE=1 \                # 禁用IB
  -e NCCL_DEBUG=INFO \                  # 调试日志
  -e HSA_OVERRIDE_GFX_VERSION=12.0.1 \  # GPU版本
  rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.8.0
```

### 网络接口名称查找

```bash
# 查看可用网络接口
ip addr show

# 常见接口名：
# - eth0, eth1 (以太网)
# - eno1, enp0s1 (板载网卡)
# - wlan0 (无线)

# 示例输出：
# 2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP>
#     inet 192.168.1.100/24
```

### 防火墙配置

如果启用了防火墙，需要开放 DDP 端口：

```bash
# Ubuntu/Debian
sudo ufw allow 29500/tcp
sudo ufw reload

# 或临时关闭防火墙（不推荐）
sudo ufw disable
```

---

## 🔍 故障排查

### 常见问题速查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 容器无法连接 | 网络/防火墙 | 检查ping、开放端口 |
| GPU 不可用 | 设备映射错误 | 检查 /dev/kfd、/dev/dri |
| 性能很慢 | 网络接口错误 | 使用有线网卡 |
| 依赖安装失败 | 网络问题 | 检查容器网络连接 |

### 问题1: 如何查看我的网络接口名？

```bash
ip addr show
# 或
ifconfig
```

查找带有IP地址的接口，通常是：
- `eth0`, `eth1` - 以太网
- `eno1`, `enp0s1` - 板载网卡
- `wlan0` - 无线网卡

### 问题2: 容器无法相互通信

**症状：**
```
RuntimeError: Connection refused
或
Timeout waiting for connection
```

**排查步骤：**

1. 检查网络连通性
```bash
# 在容器内
ping 192.168.1.101

# 检查端口
telnet 192.168.1.100 29500
```

2. 检查 NCCL 日志
```bash
# 容器启动时设置
export NCCL_DEBUG=INFO

# 查看日志，应该看到：
# NCCL INFO NET/Socket : Using [0]eth0:192.168.1.100<0>
```

3. 检查防火墙
```bash
sudo ufw status
sudo ufw allow 29500
```

4. 确保两个节点几乎同时启动（30秒内）

### 问题3: GPU 不可用

**症状：**
```python
torch.cuda.is_available() = False
```

**解决方案：**

1. 确认主机 GPU 可用
```bash
rocm-smi
```

2. 检查 Docker 设备映射
```bash
# 容器内
ls -l /dev/kfd /dev/dri
```

3. 检查环境变量
```bash
echo $HSA_OVERRIDE_GFX_VERSION
echo $PYTORCH_ROCM_ARCH
```

4. 在容器内测试
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

### 问题4: 性能很慢

**可能原因：**

1. 使用了错误的网络接口
```bash
# 检查 NCCL 日志，确认接口
export NCCL_DEBUG=INFO
# 应该看到正确的网卡IP
```

2. 梯度累积步数太小
```bash
# 增加梯度累积
--gradient_accumulation_steps 16
```

3. 没有使用混合精度
```bash
# 启用 BF16 或 FP16
--bf16
```

4. 使用了无线网络
```bash
# 改用有线网络接口
./docker_run_ddp.sh 0 192.168.1.100 tiny 5 eno1
```

---

## 📊 性能优化建议

### 1. 网络优化

```bash
# 对于 1Gbps 网络
GRADIENT_ACCUMULATION_STEPS=8    # 减少通信频率

# 对于 10Gbps 网络
GRADIENT_ACCUMULATION_STEPS=4    # 可以更频繁通信
```

### 2. 批次大小调优

```bash
# Tiny 模型 (50M)
BATCH_SIZE=16
GRAD_ACCUM=8
# 有效批次 = 16 × 2 × 8 = 256

# Small 模型 (117M)  
BATCH_SIZE=8
GRAD_ACCUM=16
# 有效批次 = 8 × 2 × 16 = 256

# Medium 模型 (345M)
BATCH_SIZE=4
GRAD_ACCUM=32
# 有效批次 = 4 × 2 × 32 = 256
```

### 3. NCCL 缓冲区优化

```bash
export NCCL_BUFFSIZE=2097152     # 2MB缓冲区
export NCCL_P2P_LEVEL=NVL        # P2P传输级别
```

### 4. 监控工具

**在容器外另开终端：**

```bash
# 监控GPU
watch -n 1 rocm-smi

# 查看容器日志
docker logs -f gpt-train-node0  # 主节点
docker logs -f gpt-train-node1  # 从节点
```

**在容器内（如果选择了交互式模式）：**

```bash
# 会自动显示进度条和训练指标
```

---

## 💡 最佳实践

### 1. 首次使用建议

建议先运行网络测试：
```bash
./test_docker_ddp_network.sh <主节点IP>
```

### 2. 使用同一 Docker 镜像

确保两个节点使用相同版本的镜像：
```bash
IMAGE=rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.8.0
```

### 3. 代码同步策略

**选项A: NFS 共享（推荐）**
```bash
# 主节点
sudo apt install nfs-kernel-server
echo "/home/user/gpt_train *(rw,sync,no_subtree_check)" >> /etc/exports
sudo exportfs -a

# 从节点
sudo apt install nfs-common
sudo mount 192.168.1.100:/home/user/gpt_train /home/user/gpt_train
```

**选项B: Git 同步**
```bash
# 每次训练前
git pull
```

**选项C: rsync 同步**
```bash
rsync -avz /path/to/gpt_train/ user@node2:/path/to/gpt_train/
```

### 4. 网络接口选择

优先使用有线网络接口（eth0, eno1）而不是无线（wlan0）。

查看接口速度：
```bash
ethtool eth0 | grep Speed
# Speed: 1000Mb/s
```

### 5. 时间同步

```bash
# 安装 NTP
sudo apt install -y ntp
sudo systemctl start ntp

# 验证时间同步
timedatectl status
```

### 6. 日志管理

```bash
# 保存训练日志
docker logs gpt-train-node0 > train_node0.log 2>&1

# 实时查看
docker logs -f gpt-train-node0

# 查看最后100行
docker logs --tail 100 gpt-train-node0
```

### 7. 防火墙最佳实践

训练前开放端口：
```bash
sudo ufw allow 29500/tcp
sudo ufw status numbered
```

训练后可以关闭（可选）：
```bash
sudo ufw delete <rule-number>
```

### 8. 网络带宽影响

| 网络类型 | 带宽 | 通信开销 | 推荐配置 |
|---------|------|---------|---------|
| 千兆以太网 | 1Gbps | 20-30% | 梯度累积8-16 |
| 万兆以太网 | 10Gbps | 10-15% | 梯度累积4-8 |
| InfiniBand | 40Gbps+ | <5% | 梯度累积2-4 |

---

## 🎯 完整工作流程

### 一次性准备（每个节点）

```bash
# 1. 安装 Docker
sudo apt update
sudo apt install -y docker.io
sudo usermod -aG docker $USER
# 重新登录

# 2. 验证 GPU
rocm-smi

# 3. 拉取镜像
docker pull rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.8.0

# 4. 准备代码
cd /home/user
git clone <your-repo> gpt_train
cd gpt_train

# 5. 赋予执行权限
chmod +x docker_run_ddp.sh test_docker_ddp_network.sh

# 6. 配置防火墙
sudo ufw allow 29500/tcp
```

### 每次训练流程

```bash
# 节点1 (主节点)
cd ~/gpt_train

# 可选：测试网络
./test_docker_ddp_network.sh 192.168.1.100

# 启动训练
./docker_run_ddp.sh 0 192.168.1.100

# 节点2 (从节点) - 几乎同时运行
cd ~/gpt_train

# 可选：测试网络
./test_docker_ddp_network.sh 192.168.1.100

# 启动训练
./docker_run_ddp.sh 1 192.168.1.100

# 容器内会自动启动训练
# 或手动运行 train_multi_gpu.py
```

### 训练中监控

```bash
# 在主节点容器外，另一个终端
watch -n 1 rocm-smi

# 查看日志
docker logs -f gpt-train-node0
```

### 训练完成后

```bash
# 查看模型文件
ls -lh gpt_model_docker_ddp_tiny/

# 查看输出
ls -lh output_docker_ddp_tiny/

# 清理容器（可选）
docker stop gpt-train-node0 gpt-train-node1
docker rm gpt-train-node0 gpt-train-node1
```

---

## ✅ 检查清单

### 启动前确认

- [ ] 两个节点都安装了Docker
- [ ] 两个节点都有GPU和ROCm驱动
- [ ] 两个节点网络可以ping通
- [ ] 防火墙允许29500端口
- [ ] 代码在两个节点的相同路径
- [ ] 知道正确的网络接口名
- [ ] 脚本有执行权限
- [ ] Docker 镜像版本一致
- [ ] 时间已同步（可选但推荐）

### 训练中检查

- [ ] 两个容器都成功启动
- [ ] NCCL日志显示正确的网络接口
- [ ] GPU利用率正常（通过rocm-smi）
- [ ] 没有连接错误或超时
- [ ] 训练损失在下降

---

## 🔗 相关文档

- [双节点训练指南](TWO_NODES_TRAINING.md)
- [网络带宽分析](NETWORK_BANDWIDTH_ANALYSIS.md)
- [Docker 设置指南](DOCKER_SETUP.md)
- [NFS Docker 修复](NFS_DOCKER_FIX.md)
- [性能优化指南](PERFORMANCE_OPTIMIZATION_GUIDE.md)

---

## 🎉 总结

### 使用 Docker 运行跨节点 DDP 的关键要素

1. ✅ **使用 host 网络模式**：简化网络配置，提升性能
2. ✅ **正确映射 GPU 设备**：确保容器可访问 GPU
3. ✅ **设置 DDP 环境变量**：MASTER_ADDR, RANK, WORLD_SIZE
4. ✅ **配置 NCCL 网络接口**：使用正确的有线网卡
5. ✅ **优化梯度累积**：根据网络带宽调整通信频率
6. ✅ **同步启动容器**：在30秒内启动所有节点
7. ✅ **监控训练过程**：使用 rocm-smi 和 docker logs

### 立即开始

```bash
# 主节点
./docker_run_ddp.sh 0 <主节点IP>

# 从节点
./docker_run_ddp.sh 1 <主节点IP>
```
---



祝训练顺利！
