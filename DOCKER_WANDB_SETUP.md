# Docker 中使用 WandB 完整指南

## 🎯 目标

在 Docker 容器中使用 Weights & Biases 追踪训练过程。

## 📦 准备工作

### 步骤 1: 在宿主机上设置 WandB（一次性）

```bash
# 1. 安装 wandb
pip install wandb

# 2. 登录（会创建配置文件）
wandb login

# 输入你的 API Key（从 https://wandb.ai/authorize 获取）
```

这会在 `~/.config/wandb/` 和 `~/.netrc` 创建配置文件。

### 步骤 2: 重新构建 Docker 镜像（包含 wandb）

```bash
cd /path/to/gpt_train
./build_docker_image.sh
```

新镜像会预装 wandb 包。

## 🚀 使用方法

### 方法 1: 自动挂载配置（推荐）

脚本 `docker_run_ddp_custom.sh` 会自动挂载 WandB 配置：

```bash
# 直接使用，WandB 配置自动挂载
./docker_run_ddp_custom.sh 0 10.161.176.100
```

脚本会自动检测并挂载：
- `~/.netrc` → 容器内 `/root/.netrc`
- `~/.config/wandb/` → 容器内 `/root/.config/wandb/`

### 方法 2: 使用 API Key 环境变量

```bash
# 设置环境变量
export WANDB_API_KEY=<your-api-key>

# 启动容器（脚本会自动传递）
./docker_run_ddp_custom.sh 0 10.161.176.100
```

### 方法 3: 容器内手动登录

```bash
# 1. 启动容器
./docker_run_ddp_custom.sh 0 10.161.176.100

# 2. 在容器内登录
wandb login <your-api-key>

# 3. 运行训练（带 WandB）
python3 train_single_gpu.py \
    --model_size tiny \
    --wandb_project gpt-training
```

## 📝 完整训练示例

### 单 GPU 训练 + WandB（使用脚本）

```bash
# 1. 在宿主机登录 WandB（一次性）
wandb login

# 2. 启动容器（WandB 配置自动挂载）
./docker_run.sh

# 3. 容器内安装依赖（首次）
pip3 install transformers datasets accelerate tensorboard tqdm wandb

# 4. 在容器内训练
python3 train_single_gpu.py \
    --model_size tiny \
    --use_chinese \
    --epochs 5 \
    --batch_size 32 \
    --bf16 \
    --wandb_project "gpt-docker-training" \
    --wandb_run_name "tiny-baseline"
```

### 单 GPU 训练 + WandB（使用自定义镜像，推荐）

```bash
# 1. 在宿主机登录 WandB（一次性）
wandb login

# 2. 构建包含 wandb 的镜像（一次性）
./build_docker_image.sh

# 3. 启动容器（使用自定义镜像）
docker run -it --rm \
    --name gpt-train \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    --ipc=host \
    --shm-size=8G \
    -v $(pwd):/workspace \
    -v ~/.config/wandb:/root/.config/wandb \
    -v ~/.netrc:/root/.netrc:ro \
    -e HSA_OVERRIDE_GFX_VERSION=12.0.1 \
    -e PYTORCH_ROCM_ARCH=gfx1201 \
    gpt-train-rocm:latest

# 4. 在容器内训练（无需安装依赖）
python3 train_single_gpu.py \
    --model_size tiny \
    --use_chinese \
    --epochs 5 \
    --batch_size 32 \
    --bf16 \
    --wandb_project "gpt-docker-training" \
    --wandb_run_name "tiny-baseline"
```

### 跨节点 DDP 训练 + WandB

**主节点：**

```bash
# 1. 设置 WandB API Key
export WANDB_API_KEY=<your-api-key>

# 2. 启动容器（自动挂载配置）
./docker_run_ddp_custom.sh 0 10.161.176.100

# 3. 容器启动后会自动开始训练
# 如果需要使用 WandB，按 Ctrl+C 停止自动训练，然后：
torchrun \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=10.161.176.100 \
    --master_port=29500 \
    train_multi_gpu.py \
    --model_size tiny \
    --use_chinese \
    --epochs 5 \
    --batch_size 32 \
    --gradient_accumulation_steps 2 \
    --bf16 \
    --wandb_project "gpt-ddp-training" \
    --wandb_run_name "2nodes-tiny"
```

**从节点：**

```bash
# 使用相同配置
export WANDB_API_KEY=<your-api-key>
./docker_run_ddp_custom.sh 1 10.161.176.100

# 然后运行相同的训练命令（rank 改为 1）
torchrun \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=10.161.176.100 \
    --master_port=29500 \
    train_multi_gpu.py \
    --model_size tiny \
    --use_chinese \
    --epochs 5 \
    --batch_size 32 \
    --gradient_accumulation_steps 2 \
    --bf16 \
    --wandb_project "gpt-ddp-training" \
    --wandb_run_name "2nodes-tiny"
```

## 🔑 获取 WandB API Key

### 方法 1: 网页获取

1. 访问 https://wandb.ai
2. 登录账号
3. 访问 https://wandb.ai/authorize
4. 复制 API Key

### 方法 2: 命令行获取

```bash
# 在宿主机上
wandb login
# 按提示操作，会显示 API Key
```

## 🔧 三种配置方式对比

### 方式 1: 挂载配置文件（推荐）

**优点：**
- ✅ 一次配置，永久使用
- ✅ 安全（不暴露 API Key）
- ✅ 脚本自动处理

**设置：**
```bash
# 宿主机登录（一次性）
wandb login

# 使用脚本启动（自动挂载）
./docker_run_ddp_custom.sh 0 <IP>
```

### 方式 2: 环境变量

**优点：**
- ✅ 简单直接
- ✅ 适合 CI/CD

**缺点：**
- ⚠️ API Key 可能暴露在命令历史中

**设置：**
```bash
export WANDB_API_KEY=<your-key>
./docker_run_ddp_custom.sh 0 <IP>
```

### 方式 3: 容器内登录

**优点：**
- ✅ 灵活
- ✅ 独立配置

**缺点：**
- ⚠️ 每次启动容器都需要登录

**设置：**
```bash
# 容器内
wandb login <your-api-key>
```

## 🎯 验证 WandB 配置

### 在容器内检查

```bash
# 检查 wandb 是否安装
python3 -c "import wandb; print(f'WandB version: {wandb.__version__}')"

# 检查登录状态
wandb status

# 测试 WandB 连接
python3 -c "import wandb; wandb.init(project='test'); wandb.finish()"
```

如果成功，会显示：
```
✓ WandB 已初始化
  项目: test
  查看: https://wandb.ai/<username>/test/runs/...
```

## 📊 WandB 在 DDP 训练中的行为

### 重要提示

**只有 Rank 0（主节点）会记录到 WandB！**

这是因为：
- 所有节点的指标是相同的
- 避免重复记录
- 节省资源

### 验证

在 WandB 仪表板中，您会看到：
- 1 个运行记录（主节点）
- 包含所有节点的聚合指标
- 训练日志来自主节点

## 🔍 故障排查

### 问题 1: 容器内 wandb 未安装

```bash
⚠️  wandb未安装，如需使用请运行: pip install wandb
```

**解决：**
```bash
# 重新构建镜像（已包含 wandb）
./build_docker_image.sh
```

### 问题 2: 容器内未登录

```bash
wandb: ERROR Unable to authenticate
```

**解决方案 A: 挂载配置**
```bash
# 确保宿主机已登录
wandb login

# 启动容器（脚本会自动挂载）
./docker_run_ddp_custom.sh 0 <IP>
```

**解决方案 B: 使用 API Key**
```bash
export WANDB_API_KEY=<your-key>
./docker_run_ddp_custom.sh 0 <IP>
```

**解决方案 C: 容器内登录**
```bash
# 进入容器后
wandb login <your-api-key>
```

### 问题 3: 网络连接问题

如果容器无法访问 WandB 服务器：

```bash
# 使用离线模式
export WANDB_MODE=offline

# 训练后手动同步
wandb sync output_dir/wandb/latest-run
```

### 问题 4: 权限问题

```bash
# 确保挂载目录可访问
ls -la ~/.config/wandb
ls -la ~/.netrc

# 如果权限不对
chmod 600 ~/.netrc
chmod -R 755 ~/.config/wandb
```

## 💡 最佳实践

### 1. 使用项目和运行名称

```bash
python3 train_single_gpu.py \
    --wandb_project "gpt-production" \
    --wandb_run_name "tiny-v1.0-$(date +%Y%m%d-%H%M)"
```

### 2. 组织实验

```bash
# 开发实验
--wandb_project "gpt-dev"

# 生产训练
--wandb_project "gpt-production"

# 消融研究
--wandb_project "gpt-ablation"
```

### 3. 使用离线模式（可选）

如果网络不稳定：

```bash
# 离线训练
export WANDB_MODE=offline
python3 train_single_gpu.py --wandb_project test

# 训练完成后同步
wandb sync output_single/wandb/latest-run
```

### 4. 安全存储 API Key

```bash
# 不要在脚本中硬编码 API Key
# 使用环境变量或配置文件

# .bashrc 中设置（推荐）
echo 'export WANDB_API_KEY=<your-key>' >> ~/.bashrc
source ~/.bashrc
```

## 🎯 完整工作流程

### 一次性设置

```bash
# 1. 注册 WandB 账号
# 访问 https://wandb.ai

# 2. 在宿主机安装并登录
pip install wandb
wandb login

# 3. 构建包含 wandb 的 Docker 镜像
cd /path/to/gpt_train
./build_docker_image.sh
```

### 每次训练

```bash
# 启动容器（WandB 配置自动挂载）
./docker_run_ddp_custom.sh 0 10.161.176.100

# 容器内训练（带 WandB）
python3 train_single_gpu.py \
    --model_size tiny \
    --use_chinese \
    --wandb_project my-project \
    --wandb_run_name "exp-$(date +%Y%m%d)"
```

## 📈 查看训练结果

### 实时查看

训练开始后，终端会显示：
```
✓ WandB 已初始化 - 项目: my-project
  运行名称: exp-20250112
  查看训练: https://wandb.ai/<username>/my-project/runs/xxx
```

点击链接即可在浏览器中实时查看训练进度。

### 离线查看

训练完成后：
```bash
# 在浏览器打开
https://wandb.ai/<your-username>/<project-name>
```

## 🎉 总结

### 使用 Docker + WandB 的优势

1. ✅ **环境隔离** - Docker 提供一致环境
2. ✅ **云端追踪** - WandB 永久保存所有指标
3. ✅ **随时随地查看** - 任何设备访问训练进度
4. ✅ **自动配置** - 脚本处理所有挂载

### 快速启动命令

```bash
# 1. 宿主机登录 WandB（一次性）
wandb login

# 2. 构建镜像（包含 wandb）
./build_docker_image.sh

# 3. 启动训练
./docker_run_ddp_custom.sh 0 <IP>

# 4. 容器内使用 WandB
python3 train_single_gpu.py \
    --wandb_project my-project \
    --model_size tiny \
    --use_chinese
```

就这么简单！🚀
