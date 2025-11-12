# 自定义 Docker 镜像使用指南

## 🎯 概述

本指南介绍如何构建和使用预装所有依赖的自定义 Docker 镜像，以简化跨节点 DDP 训练的部署。

## 📦 自定义镜像的优势

### 相比原始镜像

| 特性 | 原始镜像 | 自定义镜像 |
|------|---------|-----------|
| Python 依赖 | 每次启动安装 | 预装完成 ✅ |
| 启动时间 | 较慢（需安装） | 快速 ✅ |
| 环境变量 | 手动配置 | 预配置 ✅ |
| 网络接口 | 需手动指定 | 自动检测 ✅ |
| 跨节点部署 | 每个节点都要安装 | 一次构建，到处运行 ✅ |

## 🚀 快速开始（3步）

### 步骤1: 构建自定义镜像（一次性）

```bash
cd /path/to/gpt_train
./build_docker_image.sh
```

**构建过程：**
- 基于 `rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.8.0`
- 安装系统工具（git, vim, network tools）
- 安装 Python 依赖（transformers, datasets, accelerate 等）
- 预配置 GPU 环境变量
- 预配置 NCCL/RCCL 设置

**预期时间：** 5-10分钟（取决于网络速度）

### 步骤2: 使用自定义镜像启动训练

**主节点：**
```bash
./docker_run_ddp_custom.sh 0 <主节点IP>
```

**从节点：**
```bash
./docker_run_ddp_custom.sh 1 <主节点IP>
```

### 步骤3: 等待训练完成

容器会自动：
1. ✅ 检测并配置网络接口
2. ✅ 验证 GPU 状态
3. ✅ 启动训练（无需安装依赖）

## 📋 详细说明

### Dockerfile 内容

```dockerfile
FROM rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.8.0

# 安装的系统工具
- git, wget, curl
- vim
- net-tools, iputils-ping, telnet
- htop

# 安装的 Python 包
- transformers
- datasets
- accelerate
- tensorboard
- tqdm
- sentencepiece
- protobuf

# 预配置环境变量
- HSA_OVERRIDE_GFX_VERSION=12.0.1
- PYTORCH_ROCM_ARCH=gfx1201
- NCCL_IB_DISABLE=1
- HF_ENDPOINT=https://hf-mirror.com
```

### 构建命令详解

```bash
# 查看帮助
./build_docker_image.sh

# 自动构建
# 会提示确认，然后开始构建
```

构建完成后会显示：
```
✅ 镜像构建完成！
镜像名称: gpt-train-rocm:latest
```

### 使用自定义镜像

```bash
# 基本用法（自动检测网络接口）
./docker_run_ddp_custom.sh 0 10.161.176.100

# 指定网络接口
./docker_run_ddp_custom.sh 0 10.161.176.100 tiny 5 eno1

# 训练不同模型
./docker_run_ddp_custom.sh 0 10.161.176.100 small 10
```

## 🌐 跨节点部署

### 方法1: 在每个节点构建（推荐用于开发）

在每个节点上：
```bash
cd /path/to/gpt_train
./build_docker_image.sh
```

### 方法2: 推送到 Docker Hub（推荐用于生产）

**在主节点上：**

```bash
# 1. 构建镜像
./build_docker_image.sh

# 2. 登录 Docker Hub
docker login

# 3. 打标签
docker tag gpt-train-rocm:latest <your-username>/gpt-train-rocm:latest

# 4. 推送
docker push <your-username>/gpt-train-rocm:latest
```

**在其他节点上：**

```bash
# 拉取镜像
docker pull <your-username>/gpt-train-rocm:latest

# 重新打标签（可选）
docker tag <your-username>/gpt-train-rocm:latest gpt-train-rocm:latest
```

### 方法3: 使用私有 Docker Registry

```bash
# 设置私有 registry（例如在主节点）
docker run -d -p 5000:5000 --name registry registry:2

# 打标签并推送
docker tag gpt-train-rocm:latest localhost:5000/gpt-train-rocm:latest
docker push localhost:5000/gpt-train-rocm:latest

# 在其他节点拉取
docker pull <registry-host>:5000/gpt-train-rocm:latest
docker tag <registry-host>:5000/gpt-train-rocm:latest gpt-train-rocm:latest
```

## 🔧 自定义修改

### 添加更多 Python 包

编辑 `Dockerfile`：

```dockerfile
# 在 RUN pip3 install 部分添加
RUN pip3 install --no-cache-dir \
    transformers \
    datasets \
    accelerate \
    tensorboard \
    tqdm \
    sentencepiece \
    protobuf \
    your-package-1 \
    your-package-2
```

然后重新构建：
```bash
./build_docker_image.sh
```

### 修改环境变量

编辑 `Dockerfile` 中的 ENV 部分：

```dockerfile
ENV YOUR_VAR=value \
    ANOTHER_VAR=value
```

### 添加配置文件

编辑 `Dockerfile`：

```dockerfile
# 复制配置文件
COPY your-config.yaml /workspace/config.yaml
```

## 📊 性能对比

### 首次启动时间

| 镜像类型 | 启动到训练 | 说明 |
|---------|-----------|------|
| 原始镜像 | ~2-3分钟 | 需安装依赖 |
| 自定义镜像 | ~10秒 | 依赖已预装 ✅ |

### 镜像大小

| 镜像 | 大小 |
|------|------|
| 原始镜像 | ~10 GB |
| 自定义镜像 | ~11 GB |
| 增加 | ~1 GB（Python 包） |

## 🔍 故障排查

### 问题1: 镜像构建失败

```bash
# 检查 Docker 服务
systemctl status docker

# 清理旧的构建缓存
docker system prune -a

# 重新构建
./build_docker_image.sh
```

### 问题2: 找不到自定义镜像

```bash
# 查看本地镜像
docker images | grep gpt-train-rocm

# 如果不存在，重新构建
./build_docker_image.sh
```

### 问题3: 需要更新镜像

```bash
# 删除旧镜像
docker rmi gpt-train-rocm:latest

# 重新构建
./build_docker_image.sh
```

### 问题4: 网络接口问题

自定义镜像会自动检测网络接口，但如果需要手动指定：

```bash
# 查看可用接口
ip -br addr show

# 指定接口启动
./docker_run_ddp_custom.sh 0 10.161.176.100 tiny 5 eno1
```

## 💡 最佳实践

### 1. 版本管理

为不同版本打标签：

```bash
docker tag gpt-train-rocm:latest gpt-train-rocm:v1.0
docker tag gpt-train-rocm:latest gpt-train-rocm:stable
```

### 2. 定期更新

定期更新 Python 包：

```bash
# 修改 Dockerfile，更新包版本
# 例如：pip3 install transformers==4.35.0

# 重新构建
./build_docker_image.sh
```

### 3. 多环境支持

为不同环境创建不同镜像：

```bash
# 开发环境
docker build -t gpt-train-rocm:dev -f Dockerfile.dev .

# 生产环境
docker build -t gpt-train-rocm:prod -f Dockerfile.prod .
```

### 4. 清理未使用镜像

```bash
# 清理悬空镜像
docker image prune

# 清理所有未使用镜像
docker image prune -a
```

## 📈 使用场景对比

### 何时使用原始镜像

- ✅ 快速测试
- ✅ 依赖经常变化
- ✅ 只有一个节点

使用：
```bash
./docker_run_ddp.sh 0 <IP>
```

### 何时使用自定义镜像

- ✅ 生产环境
- ✅ 多节点部署
- ✅ 依赖固定
- ✅ 需要快速启动

使用：
```bash
./docker_run_ddp_custom.sh 0 <IP>
```

## 🎯 完整工作流程

### 开发阶段

```bash
# 1. 开发和测试（使用原始镜像）
./docker_run_ddp.sh 0 <IP>

# 2. 依赖稳定后，构建自定义镜像
./build_docker_image.sh

# 3. 测试自定义镜像
./docker_run_ddp_custom.sh 0 <IP>
```

### 部署阶段

```bash
# 1. 主节点构建并推送
./build_docker_image.sh
docker tag gpt-train-rocm:latest <username>/gpt-train-rocm:v1.0
docker push <username>/gpt-train-rocm:v1.0

# 2. 其他节点拉取
docker pull <username>/gpt-train-rocm:v1.0
docker tag <username>/gpt-train-rocm:v1.0 gpt-train-rocm:latest

# 3. 所有节点启动训练
./docker_run_ddp_custom.sh 0 <IP>  # 主节点
./docker_run_ddp_custom.sh 1 <IP>  # 从节点
```

## ✅ 检查清单

构建和使用自定义镜像前：

- [ ] Docker 已安装并运行
- [ ] 有足够磁盘空间（至少 15GB）
- [ ] 网络连接良好（需下载基础镜像和包）
- [ ] Dockerfile 已按需修改
- [ ] 了解如何在节点间分发镜像

## 📚 相关文档

- [Dockerfile](Dockerfile) - 镜像定义文件
- [构建脚本](build_docker_image.sh) - 自动化构建
- [启动脚本](docker_run_ddp_custom.sh) - 使用自定义镜像
- [快速指南](DOCKER_DDP_QUICKSTART.md) - 基础使用

## 🎉 总结

**自定义镜像的核心优势：**

1. ✅ **一次构建，到处运行** - 消除环境差异
2. ✅ **快速启动** - 依赖预装，节省时间
3. ✅ **配置标准化** - 环境变量预设
4. ✅ **便于分发** - 通过 Docker Hub 或私有 Registry

**立即开始：**

```bash
# 构建镜像
./build_docker_image.sh

# 启动训练
./docker_run_ddp_custom.sh 0 <主节点IP>
