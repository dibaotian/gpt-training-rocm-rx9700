# 使用Docker训练GPT模型 - 最简单的方式

## 🐳 为什么使用Docker？

使用ROCm官方Docker镜像的优势：
- ✅ **零配置**：预装PyTorch 2.8.0 + ROCm 7.1，无需手动配置环境
- ✅ **版本匹配**：官方保证PyTorch与ROCm版本完全兼容
- ✅ **环境隔离**：不影响主机系统
- ✅ **快速启动**：几分钟即可开始训练
- ✅ **可移植**：在不同机器上重现相同环境

## ⚡ 超快速开始（3步）

```bash
# 1. 安装Docker（如果未安装）
./install_docker.sh

# 2. 重新登录系统（使docker组权限生效）

# 3. 启动训练
./docker_run.sh
```

## 📦 可用的官方镜像

### ROCm 7.1 + PyTorch 2.8.0
```bash
rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.8.0
```

查看所有镜像：https://hub.docker.com/r/rocm/pytorch/tags

---

## 🚀 快速开始（Docker方式）

### 前置步骤：安装Docker

#### 自动安装（推荐）

```bash
# 运行安装脚本（自动完成所有配置）
./install_docker.sh
```

脚本会自动：
1. ✅ 安装Docker
2. ✅ 启动Docker服务
3. ✅ 添加用户到docker组
4. ✅ 测试GPU访问

**⚠️ 重要**：安装完成后必须**重新登录系统**，docker组权限才会生效！

#### 手动安装

如果自动脚本失败，可以手动安装：

```bash
# 1. 安装Docker
sudo apt-get update
sudo apt-get install -y docker.io

# 2. 启动Docker服务
sudo systemctl start docker
sudo systemctl enable docker

# 3. 添加用户到docker组（避免每次sudo）
sudo usermod -a -G docker $USER

# 4. 重新登录系统使更改生效
# 或临时切换组：newgrp docker
```

#### 验证安装

```bash
# 检查Docker
docker --version
docker ps

# 检查GPU访问
docker run -it --rm --device=/dev/kfd --device=/dev/dri rocm/rocm-terminal rocm-smi
```

#### 常见问题

**问题：permission denied while trying to connect to Docker daemon socket**

```bash
# 原因：用户不在docker组或未重新登录
# 解决方案1：重新登录系统
# 解决方案2：临时切换组
newgrp docker

# 解决方案3：检查用户组
groups
# 应该看到 docker

# 如果没有docker组，手动添加
sudo usermod -a -G docker $USER
# 然后重新登录
```

### 方法一：使用提供的脚本（推荐）

```bash
cd gpt_train

# 1. 启动Docker容器并进入
./docker_run.sh

# 容器内已经预装好所有环境，直接训练
python3 train_single_gpu.py --model_size tiny

# 测试生成
python3 test_generation.py
```

### 方法二：手动启动Docker

```bash
cd gpt_train

# 启动容器
docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --shm-size 8G \
  -v $(pwd):/workspace \
  -w /workspace \
  rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.8.0 \
  /bin/bash

# 容器内安装额外依赖
pip3 install transformers datasets tokenizers tensorboard wandb

# 训练
python3 train_single_gpu.py --model_size tiny
```

---

## 🔧 Docker配置文件

### docker-compose.yml

```yaml
version: '3.8'

services:
  gpt-train:
    image: rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.8.0
    container_name: gpt-train-rocm
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
      - render
    cap_add:
      - SYS_PTRACE
    security_opt:
      - seccomp=unconfined
    ipc: host
    shm_size: 8G
    volumes:
      - ./:/workspace
      - ./data:/data
      - ./models:/models
    working_dir: /workspace
    environment:
      - HSA_OVERRIDE_GFX_VERSION=11.0.0
      - PYTORCH_ROCM_ARCH=gfx1100
    command: /bin/bash
    stdin_open: true
    tty: true
```

使用：
```bash
# 启动容器
docker-compose up -d

# 进入容器
docker-compose exec gpt-train bash

# 停止容器
docker-compose down
```

---

## 📋 完整工作流程

### 单GPU训练（Docker）

```bash
# 1. 启动容器
./docker_run.sh

# 2. 容器内安装依赖（首次）
pip3 install -r requirements.txt

# 3. 验证GPU
python3 -c "import torch; print(torch.cuda.is_available())"
rocm-smi

# 4. 运行训练
python3 train_single_gpu.py \
    --model_size small \
    --epochs 5 \
    --batch_size 16

# 5. 测试模型
python3 test_generation.py --model_path ./gpt_model

# 6. 退出容器
exit
```

### 多GPU分布式训练（Docker）

#### 单机多卡

```bash
# 启动容器（所有GPU）
docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --shm-size 16G \
  -v $(pwd):/workspace \
  -w /workspace \
  rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.8.0 \
  bash -c "pip3 install -r requirements.txt && torchrun --nproc_per_node=4 train_multi_gpu.py --model_size medium"
```

#### 多机多卡

每个节点运行：

**主节点（node1）：**
```bash
docker run -it --rm \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --shm-size 16G \
  -v /path/to/shared/storage:/workspace \
  -w /workspace \
  -e MASTER_ADDR=192.168.1.100 \
  -e MASTER_PORT=29500 \
  rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.8.0 \
  bash -c "pip3 install -r requirements.txt && torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=192.168.1.100 --master_port=29500 train_multi_gpu.py"
```

**从节点（node2）：**
```bash
# 同上，只需修改 --node_rank=1
```

---

## 💡 Docker vs 本地环境

| 特性 | Docker | 本地环境 |
|------|--------|---------|
| 配置难度 | ⭐ 简单 | ⭐⭐⭐ 复杂 |
| 启动速度 | 快（镜像拉取后） | 慢（首次安装） |
| 环境隔离 | ✅ 完全隔离 | ❌ 可能冲突 |
| 性能 | ~99%（几乎无损） | 100% |
| 灵活性 | 中等 | 高 |
| 调试便利性 | 中等 | 高 |

**推荐策略**：
- 🐳 **快速验证/开发**：使用Docker
- 💻 **生产训练/调优**：使用本地环境

---

## 🎯 常见Docker命令

### 基础操作
```bash
# 查看运行中的容器
docker ps

# 查看所有容器
docker ps -a

# 停止容器
docker stop <container_id>

# 删除容器
docker rm <container_id>

# 查看镜像
docker images

# 删除镜像
docker rmi <image_id>
```

### 进入运行中的容器
```bash
docker exec -it <container_name> bash
```

### 查看容器日志
```bash
docker logs <container_name>
```

### 复制文件
```bash
# 从容器复制到主机
docker cp <container>:/path/in/container /path/on/host

# 从主机复制到容器
docker cp /path/on/host <container>:/path/in/container
```

---

## 📊 资源限制

### 设置GPU数量
```bash
# 使用特定GPU
docker run --device=/dev/dri/renderD128 ...  # GPU 0
docker run --device=/dev/dri/renderD129 ...  # GPU 1
```

### 设置内存限制
```bash
docker run --shm-size=16G --memory=32G ...
```

### 设置CPU限制
```bash
docker run --cpus=8 ...
```

---

## 🔍 故障排除

### GPU不可见

```bash
# 检查主机GPU
rocm-smi

# 检查Docker能否访问GPU
docker run -it --rm --device=/dev/kfd --device=/dev/dri rocm/rocm-terminal rocm-smi

# 检查设备权限
ls -la /dev/kfd /dev/dri/
```

### 容器内验证GPU

```bash
# 进入容器后
rocm-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

### 网络问题（多节点）

```bash
# 使用host网络
docker run --network=host ...

# 或手动映射端口
docker run -p 29500:29500 ...
```

---

## 📝 最佳实践

1. **数据持久化**：使用volume挂载
   ```bash
   -v $(pwd)/data:/data \
   -v $(pwd)/models:/models \
   -v $(pwd)/output:/output
   ```

2. **共享内存**：设置足够的shm-size
   ```bash
   --shm-size 16G  # 多GPU训练建议16G+
   ```

3. **代码同步**：挂载整个项目目录
   ```bash
   -v $(pwd):/workspace -w /workspace
   ```

4. **环境变量**：传递必要的配置
   ```bash
   -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
   -e NCCL_DEBUG=INFO
   ```

---

## 🎓 参考资源

- [ROCm Docker Hub](https://hub.docker.com/r/rocm/pytorch)
- [ROCm Docker文档](https://rocm.docs.amd.com/en/latest/deploy/docker.html)
- [PyTorch Docker指南](https://pytorch.org/get-started/locally/)
