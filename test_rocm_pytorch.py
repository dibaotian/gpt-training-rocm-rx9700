#!/usr/bin/env python3
"""
ROCm PyTorch GPU 验证脚本
用于验证 ROCm 环境下的 PyTorch GPU 功能
"""

import sys
import os

def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def check_pytorch():
    """检查 PyTorch 安装"""
    print_section("PyTorch 安装检查")
    try:
        import torch
        print(f"✓ PyTorch 已安装")
        print(f"  版本: {torch.__version__}")
        return torch
    except ImportError:
        print("✗ PyTorch 未安装")
        print("  请运行: pip install torch")
        return None

def check_rocm_info(torch):
    """检查 ROCm 信息"""
    print_section("ROCm 环境信息")
    
    # 检查 HIP 版本
    if hasattr(torch.version, 'hip'):
        print(f"✓ HIP 版本: {torch.version.hip}")
    else:
        print("✗ 无法获取 HIP 版本")
    
    # 检查 CUDA API (ROCm 兼容层)
    print(f"  CUDA API 可用: {torch.cuda.is_available()}")
    
    # 环境变量
    print("\n环境变量:")
    env_vars = [
        'HSA_OVERRIDE_GFX_VERSION',
        'PYTORCH_ROCM_ARCH',
        'AMD_SERIALIZE_KERNEL',
        'GPU_MAX_HW_QUEUES',
        'HSA_ENABLE_SDMA',
        'ROCR_VISIBLE_DEVICES',
        'HIP_VISIBLE_DEVICES'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, '未设置')
        print(f"  {var}: {value}")

def check_gpu_availability(torch):
    """检查 GPU 可用性"""
    print_section("GPU 可用性检查")
    
    if not torch.cuda.is_available():
        print("✗ GPU 不可用")
        print("\n可能的原因:")
        print("  1. ROCm 驱动未正确安装")
        print("  2. Docker 容器未正确配置 GPU 设备")
        print("  3. 环境变量设置不正确")
        return False
    
    print("✓ GPU 可用")
    
    # GPU 数量
    gpu_count = torch.cuda.device_count()
    print(f"  检测到 GPU 数量: {gpu_count}")
    
    # 每个 GPU 的详细信息
    print("\nGPU 设备信息:")
    for i in range(gpu_count):
        print(f"\n  GPU {i}:")
        print(f"    名称: {torch.cuda.get_device_name(i)}")
        
        # 显存信息
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / (1024**3)  # 转换为 GB
        print(f"    总显存: {total_memory:.2f} GB")
        print(f"    计算能力: {props.major}.{props.minor}")
        print(f"    多处理器数量: {props.multi_processor_count}")
        
        # 当前显存使用
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"    已分配显存: {allocated:.2f} GB")
            print(f"    已保留显存: {reserved:.2f} GB")
    
    return True

def test_basic_operations(torch):
    """测试基本的 GPU 操作"""
    print_section("基本 GPU 操作测试")
    
    if not torch.cuda.is_available():
        print("跳过测试（GPU 不可用）")
        return False
    
    try:
        # 测试 1: 创建张量
        print("\n测试 1: 在 GPU 上创建张量")
        x = torch.randn(1000, 1000, device='cuda')
        print(f"  ✓ 成功创建张量，形状: {x.shape}")
        print(f"  设备: {x.device}")
        
        # 测试 2: 张量运算
        print("\n测试 2: GPU 张量运算")
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.matmul(x, y)
        print(f"  ✓ 矩阵乘法成功，结果形状: {z.shape}")
        
        # 测试 3: CPU-GPU 数据传输
        print("\n测试 3: CPU-GPU 数据传输")
        cpu_tensor = torch.randn(100, 100)
        gpu_tensor = cpu_tensor.to('cuda')
        back_to_cpu = gpu_tensor.cpu()
        print(f"  ✓ CPU -> GPU -> CPU 传输成功")
        
        # 测试 4: 简单的神经网络操作
        print("\n测试 4: 简单神经网络层")
        import torch.nn as nn
        layer = nn.Linear(1000, 500).cuda()
        output = layer(x)
        print(f"  ✓ 线性层前向传播成功，输出形状: {output.shape}")
        
        # 测试 5: 梯度计算
        print("\n测试 5: 自动梯度计算")
        x_grad = torch.randn(10, 10, device='cuda', requires_grad=True)
        y_grad = x_grad ** 2
        loss = y_grad.sum()
        loss.backward()
        print(f"  ✓ 反向传播成功")
        print(f"  梯度形状: {x_grad.grad.shape}")
        
        print("\n✓ 所有基本操作测试通过!")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def performance_test(torch):
    """性能测试"""
    print_section("GPU 性能测试")
    
    if not torch.cuda.is_available():
        print("跳过测试（GPU 不可用）")
        return
    
    import time
    
    # 矩阵乘法性能测试
    print("\n矩阵乘法性能测试 (2000x2000):")
    
    # CPU 测试
    print("\n  CPU 测试:")
    x_cpu = torch.randn(2000, 2000)
    y_cpu = torch.randn(2000, 2000)
    
    start = time.time()
    for _ in range(10):
        z_cpu = torch.matmul(x_cpu, y_cpu)
    cpu_time = (time.time() - start) / 10
    print(f"    平均时间: {cpu_time*1000:.2f} ms")
    
    # GPU 测试
    print("\n  GPU 测试:")
    x_gpu = torch.randn(2000, 2000, device='cuda')
    y_gpu = torch.randn(2000, 2000, device='cuda')
    
    # 预热
    for _ in range(5):
        z_gpu = torch.matmul(x_gpu, y_gpu)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(10):
        z_gpu = torch.matmul(x_gpu, y_gpu)
    torch.cuda.synchronize()
    gpu_time = (time.time() - start) / 10
    print(f"    平均时间: {gpu_time*1000:.2f} ms")
    
    # 加速比
    speedup = cpu_time / gpu_time
    print(f"\n  GPU 加速比: {speedup:.2f}x")
    
    if speedup > 1:
        print("  ✓ GPU 性能正常")
    else:
        print("  ⚠ GPU 性能异常（可能比 CPU 慢）")

def check_rocm_smi():
    """检查 rocm-smi 工具"""
    print_section("ROCm SMI 系统信息")
    
    try:
        import subprocess
        result = subprocess.run(['rocm-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("✗ rocm-smi 执行失败")
            print(result.stderr)
    except FileNotFoundError:
        print("✗ rocm-smi 未找到")
        print("  请确保 ROCm 已正确安装")
    except Exception as e:
        print(f"✗ 执行 rocm-smi 时出错: {e}")

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("  ROCm PyTorch GPU 验证工具")
    print("=" * 60)
    
    # 检查 PyTorch
    torch = check_pytorch()
    if torch is None:
        sys.exit(1)
    
    # 检查 ROCm 信息
    check_rocm_info(torch)
    
    # 检查 GPU 可用性
    gpu_available = check_gpu_availability(torch)
    
    if gpu_available:
        # 基本操作测试
        test_success = test_basic_operations(torch)
        
        if test_success:
            # 性能测试
            performance_test(torch)
    
    # ROCm SMI 信息
    check_rocm_smi()
    
    # 总结
    print_section("验证总结")
    if gpu_available:
        print("✓ ROCm PyTorch GPU 环境正常")
        print("\n可以开始使用 GPU 进行训练!")
        print("\n示例代码:")
        print("  import torch")
        print("  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
        print("  model = YourModel().to(device)")
        print("  data = data.to(device)")
    else:
        print("✗ GPU 不可用，请检查 ROCm 环境配置")
        print("\n建议:")
        print("  1. 检查 Docker 容器是否正确挂载 GPU 设备")
        print("  2. 验证 ROCm 驱动是否正确安装")
        print("  3. 检查环境变量设置")
        print("  4. 查看 Docker 启动脚本中的设备配置")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
