# GPT模型训练 - RT9700 ROCm环境

完整的GPT模型训练方案，支持单GPU和多GPU分布式训练。

## 📋 目录

- [快速开始](#快速开始)
- [环境安装](#环境安装)
- [阶段一：单GPU训练](#阶段一单gpu训练)
- [阶段二：多GPU分布式训练](#阶段二多gpu分布式训练)
- [测试模型](#测试模型)
- [常见问题](#常见问题)
- [项目文件说明](#项目文件说明)

## 🚀 快速开始

### 🐳 方式一：使用Docker（最简单，强烈推荐）

使用官方ROCm PyTorch Docker镜像，零配置开始训练！

```bash
# 1. 进入项目目录
cd gpt_train

# 2. 启动Docker容器（自动拉取镜像并配置环境）
./docker_run.sh

# 3. 容器内安装依赖（首次）
pip3 install -r requirements.txt

# 4. 验证GPU
python3 -c "import torch; print(torch.cuda.is_available())"

# 5. 运行训练
python3 train_single_gpu.py --model_size tiny

# 6. 测试生成
python3 test_generation.py
```

**Docker优势**：
- ✅ 预装PyTorch 2.8.0 + ROCm 7.1
- ✅ 版本完全匹配，无兼容性问题
- ✅ 几分钟即可开始训练
- ✅ 环境隔离，不影响主机

详细文档：[DOCKER_SETUP.md](DOCKER_SETUP.md)

### 💻 方式二：本地环境（使用uv）

如果您需要更高的灵活性或性能微调：

```bash
# 1. 进入项目目录
cd gpt_train

# 2. 安装环境（自动安装uv并配置）
./setup_env.sh

# 3. 运行训练（使用默认tiny模型，3轮训练）
chmod +x run_single_gpu.sh
./run_single_gpu.sh

# 4. 测试生成
python3 test_generation.py

# 或使用uv运行（无需激活虚拟环境）
uv run python test_generation.py
```

## 📦 环境安装

本项目推荐使用 **uv** 来管理Python环境和依赖。uv是一个极快的Python包管理器，比传统的pip和venv快10-100倍。

### 方法一：自动安装（推荐）

```bash
# 运行setup脚本，自动安装uv并配置环境
./setup_env.sh
```

### 方法二：手动安装

#### 步骤1: 安装uv

```bash
# 安装uv（如果未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 添加到PATH（如果需要）
export PATH="$HOME/.cargo/bin:$PATH"

# 验证安装
uv --version
```

#### 步骤2: 创建虚拟环境

```bash
# 使用uv创建虚拟环境（非常快！）
uv venv

# 激活虚拟环境
source .venv/bin/activate
```

#### 步骤3: 安装PyTorch (ROCm版本)

```bash
# 使用uv安装PyTorch
# ROCm 6.1
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

# ROCm 6.0
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

#### 步骤4: 安装其他依赖

```bash
# 使用uv同步依赖（从pyproject.toml）
uv pip install -r requirements.txt

# 或使用pyproject.toml
uv pip install -e .
```

#### 步骤5: 验证环境

```bash
# 检查PyTorch是否能识别GPU
python3 -c "import torch; print(f'GPU可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 检查ROCm
rocm-smi
```

### 传统方式（venv + pip）

如果您不想使用uv，也可以使用传统的venv和pip：

```bash
# 创建虚拟环境
python3 -m venv gpt_train_env
source gpt_train_env/bin/activate

# 安装PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

# 安装依赖
pip3 install -r requirements.txt
```

## 🎯 阶段一：单GPU训练

### 使用脚本快速启动

```bash
# 基础训练（tiny模型，3轮）
./run_single_gpu.sh

# 自定义参数
./run_single_gpu.sh <模型大小> <轮数> <批次大小>

# 示例：训练small模型，5轮，批次大小16
./run_single_gpu.sh small 5 16
```

### 手动启动训练

```bash
# 基础训练
python3 train_single_gpu.py \
    --model_size tiny \
    --epochs 3 \
    --batch_size 8 \
    --output_dir ./output_single \
    --model_save_dir ./gpt_model

# 使用中文数据集
python3 train_single_gpu.py \
    --model_size small \
    --epochs 5 \
    --batch_size 8 \
    --use_chinese \
    --output_dir ./output_single_chinese \
    --model_save_dir ./gpt_model_chinese
```

### 模型大小选择

| 模型大小 | 参数量 | 显存需求 | 训练时长 | 适用场景 |
|---------|--------|---------|---------|---------|
| tiny    | ~50M   | <2GB    | 快速    | 快速验证 |
| small   | ~117M  | ~3GB    | 中等    | 推荐入门 |
| medium  | ~345M  | ~8GB    | 较长    | 更好效果 |

### 监控训练

```bash
# 查看GPU使用情况
watch -n 1 rocm-smi

# 使用TensorBoard查看训练曲线
tensorboard --logdir=./output_single/logs
# 然后访问 http://localhost:6006
```

## 🌐 阶段二：多GPU分布式训练

### 前置要求

1. **多个GPU节点**（每个节点至少1张GPU）
2. **网络连通**（所有节点互相可访问）
3. **共享存储**（推荐NFS，参考 `../nfs_setup.md`）
4. **RCCL已安装**（参考 `../rccl_install/`）
5. **SSH免密登录**

### 单机多卡训练

```bash
# 使用脚本（4张GPU）
./run_multi_gpu.sh small 5 16 4

# 参数说明:
# - small: 模型大小
# - 5: 训练轮数
# - 16: 每GPU批次大小
# - 4: GPU数量
```

### 多机多卡训练

#### 主节点（node1，IP: 192.168.1.100）

```bash
./run_multi_gpu.sh small 5 16 4 2 0 192.168.1.100 29500

# 参数说明:
# - small: 模型大小
# - 5: 训练轮数  
# - 16: 每GPU批次大小
# - 4: 每节点GPU数
# - 2: 总节点数
# - 0: 当前节点rank (主节点为0)
# - 192.168.1.100: 主节点IP
# - 29500: 通信端口
```

#### 从节点（node2，在node2上执行）

```bash
./run_multi_gpu.sh small 5 16 4 2 1 192.168.1.100 29500

# 注意: node_rank改为1（从节点）
```

### 手动启动（更多控制）

```bash
# 主节点
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train_multi_gpu.py \
    --model_size medium \
    --epochs 10 \
    --batch_size 16

# 从节点
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train_multi_gpu.py \
    --model_size medium \
    --epochs 10 \
    --batch_size 16
```

### 环境变量配置

```bash
# 设置网络接口（根据实际情况调整）
export NCCL_SOCKET_IFNAME=eth0

# 如果没有InfiniBand
export NCCL_IB_DISABLE=1

# 调试信息
export NCCL_DEBUG=INFO

# 主节点信息
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
```

## 🧪 测试模型

### 基础测试

```bash
# 测试单GPU训练的模型
python3 test_generation.py --model_path ./gpt_model

# 测试分布式训练的模型
python3 test_generation.py --model_path ./gpt_model_distributed
```

### 自定义生成

```bash
# 自定义提示词和参数
python3 test_generation.py \
    --model_path ./gpt_model \
    --prompt "In a world where" \
    --max_length 150 \
    --num_return_sequences 5 \
    --temperature 0.9

# 中文生成
python3 test_generation.py \
    --model_path ./gpt_model_chinese \
    --prompt "从前有一个" \
    --max_length 100
```

## ❓ 常见问题

### 1. GPU不可用

```bash
# 检查ROCm
rocm-smi

# 检查PyTorch
python3 -c "import torch; print(torch.cuda.is_available())"

# 如果返回False，重新安装ROCm版PyTorch
pip3 uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
```

### 2. 显存不足（OOM）

```bash
# 减小批次大小
python3 train_single_gpu.py --batch_size 4

# 或使用更小的模型
python3 train_single_gpu.py --model_size tiny
```

### 3. 多节点通信失败

```bash
# 检查网络连通性
ping node2

# 检查SSH
ssh node2 "hostname"

# 测试RCCL
cd ../rccl_install/rccl_multinode_test
./rccl_mpi_test

# 检查防火墙端口
sudo ufw allow 29500
```

### 4. 数据集下载失败

```bash
# 使用镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载数据集后使用本地路径
```

### 5. 训练速度慢

- 减小`max_length`（序列长度）
- 启用混合精度（如果支持）: `--fp16`
- 增加`gradient_accumulation_steps`
- 使用更快的数据集加载器

## 📁 项目文件说明

```
gpt_train/
├── TRAINING_PLAN.md          # 详细训练计划文档
├── README.md                 # 本文件（快速开始）
├── DOCKER_SETUP.md           # Docker使用指南（推荐阅读）
├── SETUP_GUIDE.md            # 本地环境故障排除
│
├── pyproject.toml            # 项目配置（uv使用）
├── requirements.txt          # Python依赖
├── docker-compose.yml        # Docker Compose配置
├── .gitignore               # Git忽略规则
│
├── setup_env.sh              # 本地环境配置脚本（uv）
├── docker_run.sh             # Docker容器启动脚本
│
├── train_single_gpu.py       # 单GPU训练脚本
├── train_multi_gpu.py        # 多GPU分布式训练脚本
├── test_generation.py        # 文本生成测试脚本
│
├── run_single_gpu.sh         # 单GPU启动脚本
├── run_multi_gpu.sh          # 多GPU启动脚本
│
├── .venv/                    # 虚拟环境（本地方式）
├── output_single/            # 单GPU训练输出（自动创建）
├── output_distributed/       # 多GPU训练输出（自动创建）
├── gpt_model/               # 单GPU模型保存（自动创建）
└── gpt_model_distributed/   # 多GPU模型保存（自动创建）
```

## 📚 相关文档

### 本项目文档
- 🐳 [Docker使用指南](DOCKER_SETUP.md) - **推荐新手阅读**，最简单的开始方式
- 📖 [详细训练计划](TRAINING_PLAN.md) - 完整的两阶段训练方案
- 🔧 [环境故障排除](SETUP_GUIDE.md) - 本地环境问题解决方案

### 相关系统配置
- [RCCL安装和测试](../rccl_install/) - 多节点通信库
- [NFS配置](../nfs_setup.md) - 共享存储配置
- [ROCm安装](../rocm_install/) - GPU驱动安装

## 🎓 学习资源

- [PyTorch分布式训练](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [RCCL文档](https://github.com/ROCmSoftwarePlatform/rccl)
- [NanoGPT项目](https://github.com/karpathy/nanoGPT)

## 💡 使用uv的优势

1. **极快的速度**：uv比pip快10-100倍
2. **更好的依赖解析**：避免依赖冲突
3. **现代化体验**：更好的错误提示和进度显示
4. **兼容pip**：可以无缝替换pip命令

### uv常用命令

```bash
# 创建虚拟环境
uv venv

# 安装包
uv pip install <package>

# 安装requirements.txt
uv pip install -r requirements.txt

# 从pyproject.toml安装
uv pip install -e .

# 列出已安装的包
uv pip list

# 运行脚本（自动使用虚拟环境）
uv run python script.py

# 同步依赖（确保环境与配置一致）
uv pip sync requirements.txt
```

## 🔧 进阶优化

### 使用Wandb跟踪实验

```bash
# 安装wandb
pip3 install wandb

# 登录（首次使用）
wandb login

# 在训练脚本中自动启用
# 访问 https://wandb.ai 查看实验
```

### 从检查点恢复训练

训练会自动保存检查点，如果中断可以恢复：

```bash
# 检查点会保存在 output_*/checkpoint-XXX/
ls output_single/

# 自动从最新检查点恢复（Trainer会自动处理）
```

## 📝 下一步

1. ✅ 完成环境安装
2. ✅ 运行单GPU训练验证环境
3. ✅ 测试模型生成效果
4. ✅ 配置多节点网络和存储
5. ✅ 运行多GPU分布式训练
6. ✅ 对比训练效率和效果

---

**需要帮助？** 查看 [TRAINING_PLAN.md](TRAINING_PLAN.md) 获取更详细的说明。
