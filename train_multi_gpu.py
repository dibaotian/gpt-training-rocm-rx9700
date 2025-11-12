#!/usr/bin/env python3
"""
多GPU分布式GPT训练脚本 - RT9700 ROCm环境
适用于阶段二：多机多卡训练
支持DDP (Distributed Data Parallel)
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
    parser = argparse.ArgumentParser(description='多GPU分布式GPT训练')
    parser.add_argument('--model_size', type=str, default='tiny', 
                        choices=['tiny', 'small', 'medium', 'large', 'xl'],
                        help='模型大小: tiny(50M), small(117M), medium(345M), large(774M), xl(1.5B)')
    parser.add_argument('--output_dir', type=str, default='./output_distributed',
                        help='输出目录')
    parser.add_argument('--model_save_dir', type=str, default='./gpt_model_distributed',
                        help='模型保存目录')
    parser.add_argument('--dataset', type=str, default='wikitext',
                        help='数据集名称')
    parser.add_argument('--epochs', type=int, default=5,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='每个设备的批次大小')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help='梯度累积步数（双节点1Gbps网络推荐8-16）')
    parser.add_argument('--max_length', type=int, default=512,
                        help='最大序列长度')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='学习率')
    parser.add_argument('--use_chinese', action='store_true',
                        help='使用中文数据集')
    parser.add_argument('--text_file', type=str, default=None,
                        help='使用本地文本文件')
    parser.add_argument('--fp16', action='store_true',
                        help='使用FP16混合精度训练')
    parser.add_argument('--bf16', action='store_true', default=True,
                        help='使用BFloat16混合精度训练（默认启用，数值更稳定）')
    parser.add_argument('--cache_dir', type=str, default='./datasets_cache',
                        help='数据集缓存目录')
    return parser.parse_args()

def setup_distributed():
    """初始化分布式环境"""
    # PyTorch DDP会自动设置这些环境变量
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        print("未检测到分布式环境变量，使用单GPU模式")
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        # 初始化进程组
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        print(f"[Rank {rank}/{world_size}] 初始化分布式训练")
        print(f"  - Local Rank: {local_rank}")
        print(f"  - Device: {torch.cuda.get_device_name(local_rank)}")
    
    return local_rank, rank, world_size

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
        },
        'large': {
            'n_positions': 1024,
            'n_embd': 1280,
            'n_layer': 36,
            'n_head': 20,
        },
        'xl': {
            'n_positions': 1024,
            'n_embd': 1600,
            'n_layer': 48,
            'n_head': 25,
        }
    }
    return configs[model_size]

def load_and_prepare_dataset(args, tokenizer, rank):
    """加载并预处理数据集"""
    
    # 创建缓存目录
    if rank == 0:
        os.makedirs(args.cache_dir, exist_ok=True)
        print(f"数据集缓存目录: {args.cache_dir}")
    
    # 优先使用本地文本文件
    if args.text_file:
        if rank == 0:
            print(f"加载本地文本文件: {args.text_file}")
        dataset = load_dataset('text', data_files=args.text_file, split='train', cache_dir=args.cache_dir)
        if rank == 0:
            print(f"✓ 加载成功，数据量: {len(dataset)}")
    elif args.use_chinese:
        # 中文数据集 - 按优先级尝试
        if rank == 0:
            print("加载中文数据集...")
        try:
            # 中文维基百科
            if rank == 0:
                print("尝试加载中文维基百科...")
            dataset = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train[:10%]", cache_dir=args.cache_dir)
            if rank == 0:
                print(f"✓ 使用中文维基百科，数据量: {len(dataset)}")
        except Exception as e:
            if rank == 0:
                print(f"维基百科加载失败: {e}")
            try:
                # Fineweb中文教育
                if rank == 0:
                    print("尝试加载Fineweb中文教育语料...")
                dataset = load_dataset("opencsg/Fineweb-Edu-Chinese-V2.1", split="train[:5%]", cache_dir=args.cache_dir)
                if rank == 0:
                    print(f"✓ 使用Fineweb中文教育，数据量: {len(dataset)}")
            except Exception as e2:
                if rank == 0:
                    print(f"Fineweb加载失败，使用英文数据集")
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=args.cache_dir)
    else:
        # 英文数据集
        if rank == 0:
            print(f"加载数据集: {args.dataset}")
        if args.dataset == 'wikitext':
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=args.cache_dir)
        else:
            dataset = load_dataset(args.dataset, cache_dir=args.cache_dir)
    
    # 数据预处理
    def tokenize_function(examples):
        text_field = 'text' if 'text' in examples else 'content'
        return tokenizer(
            examples[text_field], 
            truncation=True, 
            max_length=args.max_length,
            padding='max_length'
        )
    
    if rank == 0:
        print("开始tokenize数据集...")
    
    # 检查dataset是否为DatasetDict
    from datasets import DatasetDict
    if isinstance(dataset, DatasetDict):
        # 如果是DatasetDict，获取第一个split的列名
        first_split = list(dataset.keys())[0]
        columns_to_remove = dataset[first_split].column_names
        if rank == 0:
            print(f"数据集包含以下split: {list(dataset.keys())}")
            print(f"要删除的列: {columns_to_remove}")
    else:
        # 如果是单个Dataset，直接获取列名
        columns_to_remove = dataset.column_names
        if rank == 0:
            print(f"要删除的列: {columns_to_remove}")
    
    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=columns_to_remove
    )
    
    return tokenized_datasets

def main():
    args = parse_args()
    
    # 初始化分布式环境
    local_rank, rank, world_size = setup_distributed()
    
    # 仅在主进程打印信息
    is_main_process = (rank == 0)
    
    if is_main_process:
        print("\n" + "="*60)
        print(f"分布式训练配置")
        print("="*60)
        print(f"世界大小 (总GPU数): {world_size}")
        print(f"模型大小: {args.model_size}")
        print(f"批次大小/GPU: {args.batch_size}")
        print(f"有效批次大小: {args.batch_size * world_size * args.gradient_accumulation_steps}")
        print(f"梯度累积步数: {args.gradient_accumulation_steps}")
        print("="*60 + "\n")
    
    # 创建输出目录
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.model_save_dir, exist_ok=True)
    
    # 等待主进程创建目录
    if world_size > 1:
        dist.barrier()
    
    # 加载tokenizer
    if is_main_process:
        print("加载tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 创建模型配置
    model_config_params = get_model_config(args.model_size)
    if is_main_process:
        print(f"模型配置: {args.model_size}")
        print(f"  - 层数: {model_config_params['n_layer']}")
        print(f"  - 嵌入维度: {model_config_params['n_embd']}")
        print(f"  - 注意力头: {model_config_params['n_head']}")
    
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        **model_config_params
    )
    
    # 创建模型
    if is_main_process:
        print("创建模型...")
    model = GPT2LMHeadModel(config)
    
    # 计算参数量
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {total_params/1e6:.2f}M")
        print(f"预估显存需求: ~{total_params * 4 / 1e9:.2f}GB (FP32)")
    
    # 加载数据集
    tokenized_datasets = load_and_prepare_dataset(args, tokenizer, rank)
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 检查是否有validation数据集
    has_eval = False
    eval_dataset = None
    
    from datasets import DatasetDict
    if isinstance(tokenized_datasets, DatasetDict):
        if 'validation' in tokenized_datasets:
            has_eval = True
            eval_dataset = tokenized_datasets['validation']
            train_dataset = tokenized_datasets['train']
        else:
            train_dataset = tokenized_datasets[list(tokenized_datasets.keys())[0]]
    else:
        train_dataset = tokenized_datasets
    
    if is_main_process:
        print(f"训练数据量: {len(train_dataset)}")
        if has_eval:
            print(f"验证数据量: {len(eval_dataset)}")
    
    # 分布式训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        save_steps=500,
        save_total_limit=3,
        logging_steps=50,
        logging_dir=f"{args.output_dir}/logs",
        eval_strategy="steps" if has_eval else "no",
        eval_steps=250 if has_eval else None,
        fp16=args.fp16,
        bf16=args.bf16,
        # 分布式训练关键参数
        local_rank=local_rank,
        ddp_backend="nccl",  # 使用RCCL
        ddp_find_unused_parameters=False,
        # 其他优化
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="tensorboard" if is_main_process else "none",
        save_safetensors=True,
        # 确保只在主进程保存
        save_on_each_node=False,
    )
    
    # 打印网络带宽优化提示
    if is_main_process and world_size > 1:
        gradient_size_fp16 = (total_params * 2) / (1024**2)  # MB
        comm_per_sync = gradient_size_fp16 * 2  # All-Reduce约2倍
        sync_freq = args.gradient_accumulation_steps
        
        print("\n" + "="*60)
        print("网络通信分析")
        print("="*60)
        print(f"梯度大小(FP16): {gradient_size_fp16:.1f} MB")
        print(f"每次同步传输: {comm_per_sync:.1f} MB")
        print(f"同步频率: 每{sync_freq}步一次")
        print(f"1Gbps网络传输时间: {comm_per_sync/100:.2f}秒/次")
        print("="*60 + "\n")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    if is_main_process:
        print("\n" + "="*60)
        print("开始分布式训练...")
        print("="*60 + "\n")
    
    trainer.train()
    
    # 仅在主进程保存模型
    if is_main_process:
        print(f"\n保存模型到: {args.model_save_dir}")
        trainer.save_model(args.model_save_dir)
        tokenizer.save_pretrained(args.model_save_dir)
        
        print("\n训练完成!")
        print(f"模型保存在: {args.model_save_dir}")
        print(f"日志保存在: {args.output_dir}/logs")
        print("\n使用以下命令查看训练日志:")
        print(f"  tensorboard --logdir={args.output_dir}/logs")
    
    # 清理分布式环境
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
