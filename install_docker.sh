#!/bin/bash
# Docker安装和配置脚本

set -e

echo "=========================================="
echo "Docker安装和配置"
echo "=========================================="
echo ""

# 检查是否已安装Docker
if command -v docker &> /dev/null; then
    echo "✓ Docker已安装: $(docker --version)"
    DOCKER_INSTALLED=true
else
    echo "Docker未安装，开始安装..."
    DOCKER_INSTALLED=false
fi

# 安装Docker
if [ "$DOCKER_INSTALLED" = false ]; then
    echo ""
    echo "步骤1: 更新软件包索引"
    sudo apt-get update
    
    echo ""
    echo "步骤2: 安装必要的依赖"
    sudo apt-get install -y \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    echo ""
    echo "步骤3: 安装Docker"
    sudo apt-get install -y docker.io
    
    echo ""
    echo "✓ Docker安装完成"
fi

# 启动Docker服务
echo ""
echo "步骤4: 启动Docker服务"
sudo systemctl start docker
sudo systemctl enable docker
echo "✓ Docker服务已启动并设置为开机自启"

# 检查当前用户是否在docker组
echo ""
echo "步骤5: 配置用户权限"
if groups $USER | grep -q docker; then
    echo "✓ 用户 $USER 已在docker组"
else
    echo "添加用户 $USER 到docker组..."
    sudo usermod -a -G docker $USER
    echo "✓ 用户已添加到docker组"
    echo ""
    echo "⚠️  重要提示："
    echo "   需要重新登录或运行以下命令使更改生效:"
    echo "   newgrp docker"
    echo ""
    echo "   或者完全退出并重新登录系统"
    echo ""
    read -p "是否现在切换到docker组？(将在新shell中生效) (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        exec sg docker "$0 verify"
    fi
fi

# 验证Docker
echo ""
echo "步骤6: 验证Docker安装"
if docker ps &> /dev/null; then
    echo "✓ Docker运行正常"
    docker --version
else
    echo "⚠️  Docker权限问题"
    echo "请重新登录系统，然后运行: docker ps"
fi

# 测试AMD GPU访问
echo ""
echo "步骤7: 测试AMD GPU访问"
if [ -e /dev/kfd ] && [ -e /dev/dri ]; then
    echo "✓ 检测到AMD GPU设备"
    echo "  /dev/kfd: 存在"
    echo "  /dev/dri: 存在"
    
    # 尝试运行ROCm测试容器
    echo ""
    echo "测试Docker GPU访问..."
    if docker run --rm --device=/dev/kfd --device=/dev/dri rocm/rocm-terminal:latest rocm-smi &> /dev/null; then
        echo "✓ Docker可以访问AMD GPU"
    else
        echo "正在拉取ROCm测试镜像..."
        docker pull rocm/rocm-terminal:latest
        if docker run --rm --device=/dev/kfd --device=/dev/dri rocm/rocm-terminal:latest rocm-smi; then
            echo "✓ Docker可以访问AMD GPU"
        else
            echo "⚠️  Docker无法访问GPU，请检查设备权限"
        fi
    fi
else
    echo "⚠️  未找到AMD GPU设备"
    echo "  /dev/kfd: $([ -e /dev/kfd ] && echo '存在' || echo '不存在')"
    echo "  /dev/dri: $([ -e /dev/dri ] && echo '存在' || echo '不存在')"
fi

echo ""
echo "=========================================="
echo "Docker安装完成!"
echo "=========================================="
echo ""
echo "下一步:"
echo "  1. 如果刚添加到docker组，请重新登录系统"
echo "  2. 验证Docker: docker ps"
echo "  3. 启动训练容器: ./docker_run.sh"
echo "  4. 查看Docker使用指南: cat DOCKER_SETUP.md"
echo ""

# 显示有用的Docker命令
echo "常用Docker命令:"
echo "  docker ps                    - 查看运行中的容器"
echo "  docker images                - 查看本地镜像"
echo "  docker system df             - 查看磁盘使用情况"
echo "  docker system prune          - 清理未使用的资源"
echo ""
