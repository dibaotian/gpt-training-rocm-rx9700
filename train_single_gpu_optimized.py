#!/usr/bin/env python3
"""
单GPU GPT训练脚本 - 优化版
针对计算密集型场景优化，最大化 GPU 利用率和 VRAM 使用率
"""

import os
import torch
import torch.nn as nn
from transformers import (
    GPT2LMHeadModel, 
    GPT2Config, 
    GPT2Tokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='单GPU GPT训练 - 优化版')
    
    # 模型配置
    parser.add_argument('--model_size', type=str, default='small', 
                        choices=['tiny', 'small', 'medium'],
                        help='模型大小: tiny(50M), small(117M), medium(345M)')
    
    # 数据集配置
    parser.add_argument('--dataset', type=str, default='wikitext',
                        help='数据集名称')
    parser.add_argument('--use_chinese', action='store_true',
                        help='使用中文数据集')
    
    # 训练超参数 - 优化版默认值
    parser.add_argument('--epochs', type=int, default=3,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='每个设备的批次大小 [优化: 默认32，原来是8]')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='梯度累积步数 [优化: 有效批次=batch_size*gradient_accumulation_steps]')
    parser.add_argument('--max_length', type=int, default=512,
                        help='最大序列长度')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='学习率')
    
    # 性能优化选项
    parser.add_argument('--fp16', action='store_true',
                        help='启用混合精度训练 (FP16) [优化: 减少显存占用，加快计算]')
    parser.add_argument('--dataloader_num_workers', type=int, default=4,
                        help='数据加载器工作进程数 [优化: 加快数据加载]')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='启用梯度检查点 [优化: 减少显存占用，略微降低速度]')
    
    # 输出目录
    parser.add_argument('--output_dir', type=str, default='./output_single_optimized',
                        help='输出目录')
    parser.add_argument('--model_save_dir', type=str, default='./gpt_model_optimized',
                        help='模型保存目录')
    
    return parser.parse_args()

def get_model_config(model_size):
    """根据模型大小返回配置"""
    configs = {
        'tiny': {
            'n_positions': 512,
            'n_embd': 384,
            'n_layer': 6,
            'n_head': 6,
        },
        'small': {
            'n_positions': 1024,
            'n_embd': 768,
            'n_layer': 12,
            'n_head': 12,
        },
        'medium': {
            'n_positions': 1024,
            'n_embd': 1024,
            'n_layer': 24,
            'n_head': 16,
        }
    }
    return configs[model_size]

def load_and_prepare_dataset(args, tokenizer):
    """加载并预处理数据集"""
    print(f"加载数据集: {args.dataset}")
    
    if args.use_chinese:
        # 中文数据集
        dataset = load_dataset("shibing624/chinese-c4-corpus", split="train[:5%]")
    else:
        # 英文数据集
        if args.dataset == 'wikitext':
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        else:
            dataset = load_dataset(args.dataset)
    
    # 数据预处理
    def tokenize_function(examples):
        # 处理不同数据集的字段名
        text_field = 'text' if 'text' in examples else 'content'
        return tokenizer(
            examples[text_field], 
            truncation=True, 
            max_length=args.max_length,
            padding='max_length'
        )
    
    print("开始tokenize数据集...")
    
    # 检查dataset是否为DatasetDict
    from datasets import DatasetDict
    if isinstance(dataset, DatasetDict):
        # 如果是DatasetDict，获取第一个split的列名
        first_split = list(dataset.keys())[0]
        columns_to_remove = dataset[first_split].column_names
        print(f"数据集包含以下split: {list(dataset.keys())}")
        print(f"要删除的列: {columns_to_remove}")
    else:
        # 如果是单个Dataset，直接获取列名
        columns_to_remove = dataset.column_names
        print(f"要删除的列: {columns_to_remove}")
    
    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=columns_to_remove,
        num_proc=args.dataloader_num_workers  # 多进程加速tokenization
    )
    
    return tokenized_datasets

def print_gpu_memory_usage():
    """打印当前 GPU 显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n{'='*60}")
        print(f"GPU 显存使用情况:")
        print(f"  已分配: {allocated:.2f} GB")
        print(f"  已预留: {reserved:.2f} GB")
        print(f"  总容量: {total:.2f} GB")
        print(f"  使用率: {(reserved/total)*100:.1f}%")
        print(f"{'='*60}\n")

def main():
    args = parse_args()
    
    # 检测后端
    if torch.version.hip is not None:
        backend = "ROCm"
    elif torch.version.cuda is not None:
        backend = "CUDA"
    else:
        backend = "CPU"
    
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n" + "="*60)
    print("训练配置 - 优化版")
    print("="*60)
    print(f"使用设备: {device} ({backend})")
    
    if device == "cuda":
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU总内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print(f"\n优化参数:")
    print(f"  批次大小: {args.batch_size}")
    print(f"  梯度累积步数: {args.gradient_accumulation_steps}")
    print(f"  有效批次大小: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  混合精度训练: {'启用' if args.fp16 else '禁用'}")
    print(f"  梯度检查点: {'启用' if args.gradient_checkpointing else '禁用'}")
    print(f"  数据加载工作进程: {args.dataloader_num_workers}")
    print("="*60 + "\n")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)
    
    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 创建模型配置
    model_config_params = get_model_config(args.model_size)
    print(f"模型配置: {args.model_size}")
    print(f"  - 层数: {model_config_params['n_layer']}")
    print(f"  - 嵌入维度: {model_config_params['n_embd']}")
    print(f"  - 注意力头: {model_config_params['n_head']}")
    
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        **model_config_params
    )
    
    # 创建模型
    print("创建模型...")
    model = GPT2LMHeadModel(config)
    
    # 启用梯度检查点（如果需要）
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("✓ 已启用梯度检查点")
    
    model.to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params/1e6:.2f}M")
    print(f"可训练参数: {trainable_params/1e6:.2f}M")
    
    # 显存占用估算
    param_memory = total_params * 4 / 1e9  # FP32: 4 bytes per param
    if args.fp16:
        param_memory = total_params * 2 / 1e9  # FP16: 2 bytes per param
    print(f"预估模型显存需求: ~{param_memory:.2f}GB ({'FP16' if args.fp16 else 'FP32'})")
    
    # 打印初始显存使用
    print_gpu_memory_usage()
    
    # 加载数据集
    tokenized_datasets = load_and_prepare_dataset(args, tokenizer)
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT使用因果语言模型，不是MLM
    )
    
    # 训练参数 - 优化版
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        
        # 批次和梯度累积 - 优化重点
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # 学习率和优化器
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        
        # 保存和日志
        save_steps=500,
        save_total_limit=3,
        logging_steps=50,
        logging_dir=f"{args.output_dir}/logs",
        
        # 评估策略
        eval_strategy="steps",
        eval_steps=250,
        
        # 性能优化
        fp16=args.fp16,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        
        # 其他
        report_to="tensorboard",
        save_safetensors=True,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets if isinstance(tokenized_datasets, dict) is False else tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'] if 'validation' in tokenized_datasets else None,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("\n" + "="*60)
    print("开始优化训练...")
    print(f"有效批次大小: {args.batch_size * args.gradient_accumulation_steps}")
    print("="*60 + "\n")
    
    # 训练前的显存使用
    print("训练前显存使用:")
    print_gpu_memory_usage()
    
    trainer.train()
    
    # 训练后的显存使用
    print("\n训练后显存使用:")
    print_gpu_memory_usage()
    
    # 保存模型
    print(f"\n保存模型到: {args.model_save_dir}")
    model.save_pretrained(args.model_save_dir)
    tokenizer.save_pretrained(args.model_save_dir)
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    print(f"模型保存在: {args.model_save_dir}")
    print(f"日志保存在: {args.output_dir}/logs")
    print(f"\n使用以下命令查看训练日志:")
    print(f"  tensorboard --logdir={args.output_dir}/logs")
    print("\n优化建议:")
    print("  - 如果 VRAM 使用率仍然较低，可以进一步增大 --batch_size")
    print("  - 如果遇到 OOM 错误，可以:")
    print("    * 减小 --batch_size")
    print("    * 增大 --gradient_accumulation_steps")
    print("    * 启用 --gradient_checkpointing")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
