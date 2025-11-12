# GPT模型训练计划 - RT9700 (ROCm环境)

## 项目概述

在AMD Radeon RX 9700上训练小型GPT模型，分两个阶段进行：
- **阶段一**：单卡训练 - 验证环境、代码和基础功能
- **阶段二**：多机多卡训练 - 扩展到分布式训练

---

## 阶段一：单卡训练

### 1.1 环境准备（Docker 方式）

#### 🐳 为什么使用 Docker？

- ✅ **零配置**：预装 PyTorch 2.8.0 + ROCm 7.1，无需手动配置环境
- ✅ **版本匹配**：官方保证 PyTorch 与 ROCm 版本完全兼容
- ✅ **环境隔离**：不影响主机系统
- ✅ **快速启动**：几分钟即可开始训练

#### Docker 环境准备（3步）

**步骤 1: 安装 Docker**

```bash
# 运行安装脚本
./install_docker.sh

# ⚠️ 重要：安装完成后必须重新登录系统！
```

**步骤 2: 验证 Docker 和 GPU**

```bash
# 检查 Docker
docker --version

# 检查 GPU 访问
docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  rocm/rocm-terminal rocm-smi
```

**步骤 3: 启动训练容器**

```bash
# 进入项目目录
cd gpt_train

# 启动 Docker 容器（使用提供的脚本）
./docker_run.sh
```

#### 容器内环境

容器已预装：
- ✅ PyTorch 2.8.0 (ROCm 7.1)
- ✅ Python 3.10
- ✅ ROCm 工具链
- ✅ CUDA 兼容层

需要额外安装：
```bash
# 在容器内执行（首次启动时）
pip3 install -r requirements.txt

# requirements.txt 包含：
# transformers
# datasets
# tokenizers
# tensorboard
# wandb (可选)
```

#### 验证环境
```bash
# 在容器内执行
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
rocm-smi
```

### 1.2 模型选择

推荐从小到大逐步尝试：

| 模型 | 参数量 | 显存需求 | 训练时长(估算) | 适用场景 |
|------|--------|----------|---------------|----------|
| NanoGPT | 10M-100M | <2GB | 快速 | 学习和验证 |
| GPT-2 Small | 117M | ~2-3GB | 中等 | 生产环境入门 |
| DistilGPT-2 | 82M | ~2GB | 快速 | 轻量级应用 |
| TinyLlama | 1.1B | ~10GB | 较长 | 更强性能 |

**推荐起步**：NanoGPT (自定义配置，50M参数)

### 1.3 数据准备

#### 方案A：使用公开数据集
```python
# 示例：使用Hugging Face datasets
from datasets import load_dataset

# 英文数据集
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# 中文数据集
dataset = load_dataset("shibing624/chinese-c4-corpus", split="train[:1%]")
```

#### 方案B：自定义文本数据
```bash
# 准备文本文件 (例如 train.txt)
# 格式：每行一个句子或段落
mkdir -p data
# 将您的文本数据放入 data/train.txt
```

### 1.4 训练脚本

创建基础训练脚本 `train_single_gpu.py`：

```python
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# 配置
model_name = "gpt2"  # 或自定义配置
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型和tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 小型配置
config = GPT2Config(
    vocab_size=50257,
    n_positions=512,      # 序列长度
    n_embd=384,           # 嵌入维度
    n_layer=6,            # 层数
    n_head=6,             # 注意力头数
)
model = GPT2LMHeadModel(config)
model.to(device)

# 加载数据
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# 数据预处理
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 训练参数
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
    fp16=False,  # ROCm可能不完全支持fp16，先用fp32
    evaluation_strategy="steps",
    eval_steps=500,
)

# 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# 开始训练
trainer.train()

# 保存模型
model.save_pretrained("./gpt_model")
tokenizer.save_pretrained("./gpt_model")
```

### 1.5 执行训练（Docker 环境）

#### 基础训练

```bash
# 方法1: 使用提供的脚本（推荐）
./docker_run.sh

# 容器内执行
python3 train_single_gpu.py --model_size small

# 监控GPU（新开一个终端，在主机上执行）
watch -n 1 rocm-smi
```

#### 优化训练（推荐）

如果您遇到 GPU 利用率高但 VRAM 使用率低的情况，使用优化版本：

```bash
# 在容器内执行
./run_single_gpu_optimized.sh

# 或直接运行优化脚本
python3 train_single_gpu_optimized.py \
    --model_size small \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --fp16  # 可选：启用混合精度

# 参考 GPU_TRAINING_OPTIMIZATION.md 了解更多优化策略
```

#### 监控和调试

```bash
# 终端1: 运行训练（容器内）
python3 train_single_gpu.py

# 终端2: 监控GPU（主机上）
watch -n 1 rocm-smi

# 终端3: 查看容器日志（主机上）
docker logs -f gpt-train-container

# 查看训练日志（容器内）
tensorboard --logdir=./output_single/logs
# 然后在浏览器访问: http://localhost:6006
```

#### Docker 常用操作

```bash
# 查看运行中的容器
docker ps

# 进入已运行的容器
docker exec -it gpt-train-container bash

# 停止容器
docker stop gpt-train-container

# 重启训练
./docker_run.sh
```

### 1.6 验证和测试

```python
# 测试生成文本 test_generation.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("./gpt_model")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt_model")

prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

---

## 阶段二：多机多卡训练

### 2.1 环境要求

#### 硬件配置
- **多个节点**：每个节点至少1张AMD GPU
- **网络**：高速网络（千兆以上，建议万兆）
- **存储**：共享存储（NFS）或一致的数据副本

#### 软件依赖
```bash
# 除了阶段一的依赖，还需要：
# 1. RCCL (ROCm Communication Library)
# 2. MPI (OpenMPI 或 MPICH)
# 3. PyTorch Distributed

# 安装MPI（如果未安装）
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev

# 验证RCCL
ls /opt/rocm/lib/librccl.so
```

### 2.2 网络配置

#### NFS共享存储（推荐）
```bash
# 在主节点上设置NFS服务器
# 参考您的 nfs_setup.md 文件

# 确保所有节点可以访问：
# - 训练数据
# - 训练脚本
# - 模型保存路径
```

#### SSH免密登录
```bash
# 在所有节点之间配置SSH免密登录
ssh-keygen -t rsa
ssh-copy-id user@node2
ssh-copy-id user@node3
```

### 2.3 分布式训练策略

#### 数据并行 (DDP - Distributed Data Parallel)
最常用的分布式训练方式：
- 每个GPU持有完整模型副本
- 数据分片到不同GPU
- 梯度通过RCCL同步

#### 训练配置
```python
# train_multi_gpu.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

def setup_distributed():
    """初始化分布式环境"""
    dist.init_process_group(backend="nccl")  # RCCL backend
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank

def main():
    local_rank = setup_distributed()
    
    # 模型配置
    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
    )
    
    model = GPT2LMHeadModel(config)
    model.to(local_rank)
    
    # 包装为DDP模型
    model = DDP(model, device_ids=[local_rank])
    
    # 数据加载
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 训练参数（多GPU）
    training_args = TrainingArguments(
        output_dir="./output_distributed",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=16,  # 每个GPU的batch size
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,   # 梯度累积
        save_steps=500,
        save_total_limit=3,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=250,
        local_rank=local_rank,
        ddp_backend="nccl",  # 使用RCCL
        ddp_find_unused_parameters=False,
    )
    
    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )
    
    trainer.train()
    
    # 仅在主进程保存模型
    if local_rank == 0:
        model.module.save_pretrained("./gpt_model_distributed")
        tokenizer.save_pretrained("./gpt_model_distributed")

if __name__ == "__main__":
    main()
```

### 2.4 启动脚本

#### 方案A：使用torchrun（推荐）
```bash
#!/bin/bash
# run_distributed.sh

# 单机多卡
torchrun --nproc_per_node=4 \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr="localhost" \
         --master_port=29500 \
         train_multi_gpu.py

# 多机多卡 - 主节点
torchrun --nproc_per_node=4 \
         --nnodes=2 \
         --node_rank=0 \
         --master_addr="192.168.1.100" \
         --master_port=29500 \
         train_multi_gpu.py

# 多机多卡 - 从节点（在node2上执行）
torchrun --nproc_per_node=4 \
         --nnodes=2 \
         --node_rank=1 \
         --master_addr="192.168.1.100" \
         --master_port=29500 \
         train_multi_gpu.py
```

#### 方案B：使用MPI启动
```bash
#!/bin/bash
# run_mpi.sh

# 创建hostfile
cat > hostfile << EOF
node1 slots=4
node2 slots=4
EOF

# 使用mpirun启动
mpirun -np 8 \
       --hostfile hostfile \
       -x NCCL_DEBUG=INFO \
       -x NCCL_SOCKET_IFNAME=eth0 \
       python3 train_multi_gpu.py
```

### 2.5 环境变量配置

```bash
# 关键环境变量
export NCCL_DEBUG=INFO              # 调试信息
export NCCL_SOCKET_IFNAME=eth0      # 网络接口
export NCCL_IB_DISABLE=1            # 如果没有InfiniBand
export GLOO_SOCKET_IFNAME=eth0      # Gloo后端
export MASTER_ADDR=192.168.1.100    # 主节点IP
export MASTER_PORT=29500            # 通信端口
```

### 2.6 性能优化

#### 梯度累积
```python
# 减少通信频率，提高吞吐量
training_args = TrainingArguments(
    gradient_accumulation_steps=8,  # 累积8步再更新
    ...
)
```

#### 混合精度训练（如果支持）
```python
training_args = TrainingArguments(
    fp16=True,  # 或 bf16=True
    ...
)
```

#### 优化RCCL性能
```bash
# 使用TCP而非共享内存（多节点）
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=eth0

# 调整RCCL缓冲区
export NCCL_BUFFSIZE=2097152
```

### 2.7 监控和调试

#### 监控训练进度
```bash
# 使用tensorboard
tensorboard --logdir=./output_distributed/runs

# 或使用wandb
# 在训练脚本中添加：
# import wandb
# wandb.init(project="gpt-training")
```

#### 检查GPU利用率
```bash
# 在所有节点上运行
watch -n 1 rocm-smi

# 查看进程
ps aux | grep python
```

#### 常见问题排查
```bash
# 测试RCCL通信
cd rccl_install/rccl_multinode_test
./rccl_mpi_test

# 检查网络连通性
ping node2
ssh node2 "hostname"

# 查看RCCL日志
export NCCL_DEBUG=INFO
# 重新运行训练，查看详细日志
```

---

## 阶段三：进阶优化（可选）

### 3.1 模型并行

对于更大的模型（>1B参数），可以使用模型并行：
```python
# 使用DeepSpeed或Megatron-LM
# 将模型分片到多个GPU
```

### 3.2 FlashAttention

如果ROCm支持，可以使用FlashAttention加速：
```python
from transformers import GPT2Config

config = GPT2Config(
    use_flash_attention=True,  # 需要ROCm 5.5+
)
```

### 3.3 检查点和恢复

```python
# 自动保存检查点
training_args = TrainingArguments(
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
)

# 从检查点恢复
trainer.train(resume_from_checkpoint="./output/checkpoint-1000")
```

---

## 附录：完整工作流程

### 快速开始检查清单（Docker 环境）

#### 阶段一：单卡训练（Docker）
- [ ] 安装 Docker
  ```bash
  ./install_docker.sh
  # 重新登录系统
  ```
- [ ] 验证 Docker 和 GPU
  ```bash
  docker --version
  docker run -it --rm --device=/dev/kfd --device=/dev/dri rocm/rocm-terminal rocm-smi
  ```
- [ ] 启动训练容器
  ```bash
  cd gpt_train
  ./docker_run.sh
  ```
- [ ] 安装 Python 依赖（容器内，首次）
  ```bash
  pip3 install -r requirements.txt
  ```
- [ ] 验证 GPU 可用性（容器内）
  ```bash
  python3 -c "import torch; print(torch.cuda.is_available())"
  rocm-smi
  ```
- [ ] 运行训练脚本（容器内）
  ```bash
  # 基础训练
  python3 train_single_gpu.py --model_size tiny
  
  # 或优化训练
  ./run_single_gpu_optimized.sh
  ```
- [ ] 监控 GPU（主机新终端）
  ```bash
  watch -n 1 rocm-smi
  ```
- [ ] 验证模型生成（容器内）
  ```bash
  python3 test_generation.py
  ```

#### 阶段二：多机多卡训练（Docker）
- [ ] 在所有节点安装 Docker
- [ ] 配置多节点网络和 SSH 免密登录
- [ ] 设置 NFS 共享存储（推荐）
  - 挂载共享目录到所有节点相同路径
- [ ] 在每个节点拉取 Docker 镜像
  ```bash
  docker pull rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.8.0
  ```
- [ ] 测试节点间网络连通性
  ```bash
  ping node2
  ssh node2 "docker run --rm rocm/rocm-terminal rocm-smi"
  ```
- [ ] 启动多节点 Docker 容器
  - 主节点（node1）
    ```bash
    docker run -it --rm \
      --network=host \
      --device=/dev/kfd --device=/dev/dri \
      -v /shared/storage:/workspace \
      -e MASTER_ADDR=192.168.1.100 \
      -e MASTER_PORT=29500 \
      rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.8.0 \
      bash
    ```
  - 从节点（node2）：类似命令
- [ ] 启动分布式训练（容器内）
  ```bash
  # 主节点
  torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=192.168.1.100 --master_port=29500 \
    train_multi_gpu.py
  
  # 从节点
  torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr=192.168.1.100 --master_port=29500 \
    train_multi_gpu.py
  ```
- [ ] 监控所有节点的训练进度和 GPU 使用率
  ```bash
  # 在各节点主机上
  watch -n 1 rocm-smi
  ```

---

## 参考资源

### Docker 相关
1. **ROCm Docker Hub**：https://hub.docker.com/r/rocm/pytorch
2. **ROCm Docker 文档**：https://rocm.docs.amd.com/en/latest/deploy/docker.html
3. **DOCKER_SETUP.md**：项目中的 Docker 环境配置指南

### PyTorch 和训练
4. **PyTorch 分布式训练**：https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
5. **Hugging Face Trainer**：https://huggingface.co/docs/transformers/main_classes/trainer
6. **NanoGPT 项目**：https://github.com/karpathy/nanoGPT

### ROCm 和 RCCL
7. **RCCL 文档**：https://github.com/ROCmSoftwarePlatform/rccl
8. **ROCm 兼容性矩阵**：https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html

### 项目文档
9. **GPU_TRAINING_OPTIMIZATION.md**：GPU 训练优化指南
10. **nfs_setup.md**：NFS 共享存储配置

---

## 下一步建议

1. **立即开始**：先完成阶段一的单卡训练，验证环境
2. **逐步扩展**：单卡成功后，再进行多机多卡
3. **持续监控**：使用wandb或tensorboard跟踪训练
4. **数据为王**：准备高质量的训练数据
5. **小步快跑**：从小模型开始，逐步增加规模
