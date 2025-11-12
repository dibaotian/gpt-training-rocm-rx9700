#!/bin/bash
# Docker DDP 网络连接测试脚本

set -e

echo "=========================================="
echo "Docker DDP 网络连接测试"
echo "=========================================="
echo ""

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <master_addr> [network_interface]"
    echo ""
    echo "示例:"
    echo "  $0 192.168.1.100"
    echo "  $0 192.168.1.100 eno1"
    echo ""
    exit 1
fi

MASTER_ADDR=$1
NETWORK_INTERFACE=${2:-"eth0"}
MASTER_PORT=29500

echo "测试配置:"
echo "  主节点地址: $MASTER_ADDR"
echo "  测试端口: $MASTER_PORT"
echo "  网络接口: $NETWORK_INTERFACE"
echo ""

# 测试1: 基本网络连通性
echo "=========================================="
echo "测试1: 基本网络连通性"
echo "=========================================="
if ping -c 3 -W 2 $MASTER_ADDR &> /dev/null; then
    echo "✓ 可以ping通主节点 $MASTER_ADDR"
    
    # 显示延迟
    LATENCY=$(ping -c 3 $MASTER_ADDR | tail -1 | awk '{print $4}' | cut -d '/' -f 2)
    echo "  平均延迟: ${LATENCY}ms"
else
    echo "✗ 无法ping通主节点 $MASTER_ADDR"
    echo "  请检查:"
    echo "  1. 网络连接是否正常"
    echo "  2. IP地址是否正确"
    echo "  3. 防火墙设置"
    exit 1
fi
echo ""

# 测试2: 检查网络接口
echo "=========================================="
echo "测试2: 检查网络接口"
echo "=========================================="
if ip addr show $NETWORK_INTERFACE &> /dev/null; then
    echo "✓ 网络接口 $NETWORK_INTERFACE 存在"
    
    # 显示接口信息
    IP_ADDR=$(ip addr show $NETWORK_INTERFACE | grep 'inet ' | awk '{print $2}' | cut -d '/' -f 1)
    if [ -n "$IP_ADDR" ]; then
        echo "  本地IP: $IP_ADDR"
    else
        echo "  警告: 接口未配置IP地址"
    fi
    
    # 检查接口状态
    STATE=$(ip addr show $NETWORK_INTERFACE | grep -o 'state [A-Z]*' | awk '{print $2}')
    echo "  状态: $STATE"
    
    if [ "$STATE" != "UP" ]; then
        echo "  警告: 接口未启用"
    fi
else
    echo "✗ 网络接口 $NETWORK_INTERFACE 不存在"
    echo ""
    echo "可用的网络接口:"
    ip -br addr show | grep -v '^lo'
    echo ""
    exit 1
fi
echo ""

# 测试3: 检查端口可达性
echo "=========================================="
echo "测试3: 检查端口可达性"
echo "=========================================="
echo "尝试连接 $MASTER_ADDR:$MASTER_PORT ..."
if timeout 5 bash -c "echo > /dev/tcp/$MASTER_ADDR/$MASTER_PORT" 2>/dev/null; then
    echo "✓ 端口 $MASTER_PORT 可达"
else
    echo "⚠ 端口 $MASTER_PORT 不可达"
    echo "  这是正常的，如果主节点容器尚未启动"
    echo "  请在主节点启动容器后重新测试"
fi
echo ""

# 测试4: 检查防火墙
echo "=========================================="
echo "测试4: 检查防火墙状态"
echo "=========================================="
if command -v ufw &> /dev/null; then
    UFW_STATUS=$(sudo ufw status 2>/dev/null | head -1)
    echo "UFW状态: $UFW_STATUS"
    
    if echo "$UFW_STATUS" | grep -q "active"; then
        echo "  检查端口 $MASTER_PORT 规则..."
        if sudo ufw status | grep -q "$MASTER_PORT"; then
            echo "  ✓ 端口 $MASTER_PORT 已开放"
        else
            echo "  ⚠ 端口 $MASTER_PORT 未在防火墙规则中"
            echo ""
            echo "  建议运行:"
            echo "    sudo ufw allow $MASTER_PORT/tcp"
        fi
    fi
else
    echo "UFW未安装，跳过防火墙检查"
fi
echo ""

# 测试5: Docker检查
echo "=========================================="
echo "测试5: Docker环境检查"
echo "=========================================="
if command -v docker &> /dev/null; then
    echo "✓ Docker已安装"
    
    # 检查Docker版本
    DOCKER_VERSION=$(docker --version | awk '{print $3}' | sed 's/,//')
    echo "  版本: $DOCKER_VERSION"
    
    # 检查用户权限
    if groups | grep -q docker; then
        echo "  ✓ 当前用户在docker组"
    else
        echo "  ⚠ 当前用户不在docker组"
        echo "    建议运行: sudo usermod -aG docker $USER"
        echo "    然后重新登录"
    fi
    
    # 检查Docker服务
    if systemctl is-active --quiet docker 2>/dev/null; then
        echo "  ✓ Docker服务运行中"
    else
        echo "  ⚠ Docker服务未运行"
    fi
    
    # 检查镜像
    if docker images | grep -q "rocm/pytorch.*rocm7.1"; then
        echo "  ✓ ROCm PyTorch镜像已下载"
    else
        echo "  ⚠ ROCm PyTorch镜像未下载"
        echo "    将在首次运行时自动下载"
    fi
else
    echo "✗ Docker未安装"
    echo "  请运行: sudo apt-get install -y docker.io"
    exit 1
fi
echo ""

# 测试6: GPU检查
echo "=========================================="
echo "测试6: GPU设备检查"
echo "=========================================="
if [ -e /dev/kfd ] && [ -e /dev/dri ]; then
    echo "✓ GPU设备存在"
    echo "  /dev/kfd: ✓"
    echo "  /dev/dri: ✓"
    
    # 检查ROCm
    if command -v rocm-smi &> /dev/null; then
        echo ""
        echo "GPU信息:"
        rocm-smi --showid --showproductname 2>/dev/null | head -10 || rocm-smi | head -10
    else
        echo "  ⚠ rocm-smi未安装，无法查看GPU详情"
    fi
else
    echo "⚠ GPU设备不完整"
    echo "  /dev/kfd: $([ -e /dev/kfd ] && echo '✓' || echo '✗')"
    echo "  /dev/dri: $([ -e /dev/dri ] && echo '✓' || echo '✗')"
fi
echo ""

# 测试7: 带宽测试（可选）
echo "=========================================="
echo "测试7: 网络带宽测试（可选）"
echo "=========================================="
if command -v iperf3 &> /dev/null; then
    echo "iperf3已安装"
    echo ""
    echo "如需测试网络带宽，请执行以下步骤:"
    echo ""
    echo "1. 在主节点上启动iperf3服务器:"
    echo "   iperf3 -s"
    echo ""
    echo "2. 在从节点上运行客户端:"
    echo "   iperf3 -c $MASTER_ADDR -t 10"
    echo ""
else
    echo "iperf3未安装（可选）"
    echo "如需测试带宽，请安装: sudo apt-get install -y iperf3"
fi
echo ""

# 总结
echo "=========================================="
echo "测试总结"
echo "=========================================="
echo ""
echo "已完成的检查:"
echo "  ✓ 网络连通性"
echo "  ✓ 网络接口"
echo "  ✓ 端口可达性"
echo "  ✓ 防火墙状态"
echo "  ✓ Docker环境"
echo "  ✓ GPU设备"
echo ""
echo "下一步:"
echo "  1. 确保主节点和从节点都通过了测试"
echo "  2. 在主节点运行: ./docker_run_ddp.sh 0 $MASTER_ADDR"
echo "  3. 在从节点运行: ./docker_run_ddp.sh 1 $MASTER_ADDR"
echo ""
echo "故障排查:"
echo "  查看完整指南: DOCKER_DDP_MULTINODE_GUIDE.md"
echo ""
