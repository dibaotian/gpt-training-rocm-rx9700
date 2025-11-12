#!/usr/bin/env python3
"""
GPT模型文本生成测试脚本
用于验证训练后的模型
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='测试GPT模型文本生成')
    parser.add_argument('--model_path', type=str, default='./gpt_model',
                        help='模型路径')
    parser.add_argument('--prompt', type=str, default='Once upon a time',
                        help='输入提示文本')
    parser.add_argument('--max_length', type=int, default=100,
                        help='生成文本的最大长度')
    parser.add_argument('--num_return_sequences', type=int, default=3,
                        help='生成的文本数量')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='采样温度 (0.1-2.0)')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-K采样')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-P (nucleus) 采样')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载模型和tokenizer
    print(f"\n从 {args.model_path} 加载模型...")
    try:
        model = GPT2LMHeadModel.from_pretrained(args.model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
        model.to(device)
        model.eval()
        print("模型加载成功!")
    except Exception as e:
        print(f"错误: 无法加载模型 - {e}")
        return
    
    # 模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params/1e6:.2f}M")
    
    # 生成文本
    print(f"\n{'='*60}")
    print(f"提示文本: {args.prompt}")
    print(f"{'='*60}\n")
    
    # 编码输入
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=args.max_length,
            num_return_sequences=args.num_return_sequences,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # 解码并打印结果
    for i, output in enumerate(outputs):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"生成 #{i+1}:")
        print(f"{generated_text}")
        print(f"\n{'-'*60}\n")

if __name__ == "__main__":
    main()
