#!/bin/bash
# GPU 详细监控脚本

echo "=========================================="
echo "GPU 详细监控"
echo "=========================================="
echo ""

while true; do
    clear
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 基本信息
    echo "========== GPU 概览 =========="
    rocm-smi --showuse --showmeminfo vram --showpower --showtemp --showclocks
    
    echo ""
    echo "========== 详细计算使用率 =========="
    # 显示GPU活动百分比
    rocm-smi --showuse | grep -A 10 "GPU use"
    
    echo ""
    echo "========== 内存使用详情 =========="
    rocm-smi --showmeminfo vram | grep -E "VRAM Total|VRAM Used"
    
    echo ""
    echo "========== 进程信息 =========="
    # 显示GPU上运行的进程
    rocm-smi --showpids || echo "无法显示进程信息"
    
    echo ""
    echo "========== HIP 活动 =========="
    # 尝试显示HIP活动
    if command -v rocprof &> /dev/null; then
        echo "HIP 工具可用"
    else
        echo "HIP profiling 工具不可用"
    fi
    
    echo ""
    echo "按 Ctrl+C 退出监控"
    echo ""
    
    sleep 2
done
