#!/usr/bin/env python3
"""
单GPU GPT训练脚本 - RT9700 ROCm环境
适用于阶段一：单卡训练验证
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

# Weights & Biases 集成
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb未安装，如需使用请运行: pip install wandb")

def parse_args():
    parser = argparse.ArgumentParser(description='单GPU GPT训练')
    parser.add_argument('--model_size', type=str, default='small', 
                        choices=['tiny', 'small', 'medium', 'large', 'xl'],
                        help='模型大小: tiny(50M), small(117M), medium(345M), large(774M), xl(1.5B)')
    parser.add_argument('--output_dir', type=str, default='./output_single',
                        help='输出目录')
    parser.add_argument('--model_save_dir', type=str, default='./gpt_model',
                        help='模型保存目录')
    parser.add_argument('--dataset', type=str, default='wikitext',
                        help='数据集名称')
    parser.add_argument('--epochs', type=int, default=3,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='每个设备的批次大小')
    parser.add_argument('--max_length', type=int, default=512,
                        help='最大序列长度')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='学习率')
    parser.add_argument('--use_chinese', action='store_true',
                        help='使用中文数据集')
    parser.add_argument('--text_file', type=str, default=None,
                        help='使用本地文本文件（每行一段文本）')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='梯度累积步数（用于模拟更大批次）')
    parser.add_argument('--fp16', action='store_true',
                        help='使用混合精度训练（fp16）')
    parser.add_argument('--bf16', action='store_true',
                        help='使用BFloat16混合精度训练')
    parser.add_argument('--cache_dir', type=str, default='./datasets_cache',
                        help='数据集缓存目录')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='WandB项目名称（启用WandB追踪）')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='WandB运行名称')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='WandB团队/用户名')
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

def load_and_prepare_dataset(args, tokenizer):
    """加载并预处理数据集"""
    
    # 创建缓存目录
    os.makedirs(args.cache_dir, exist_ok=True)
    print(f"数据集缓存目录: {args.cache_dir}")
    
    # 优先使用本地文本文件
    if args.text_file:
        print(f"加载本地文本文件: {args.text_file}")
        dataset = load_dataset('text', data_files=args.text_file, split='train', cache_dir=args.cache_dir)
        print(f"✓ 加载成功，数据量: {len(dataset)}")
    elif args.use_chinese:
        # 中文数据集 - 按优先级尝试多个可用的数据集
        print("加载中文数据集...")
        try:
            # 选项1: 中文维基百科（推荐，质量高）
            print("尝试加载中文维基百科...")
            dataset = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train[:10%]", cache_dir=args.cache_dir)
            print(f"✓ 使用中文维基百科数据集，数据量: {len(dataset)}")
        except Exception as e:
            print(f"维基百科加载失败: {e}")
            try:
                # 选项2: 中文教育语料（高质量）
                print("尝试加载Fineweb中文教育语料...")
                dataset = load_dataset("opencsg/Fineweb-Edu-Chinese-V2.1", split="train[:5%]", cache_dir=args.cache_dir)
                print(f"✓ 使用Fineweb中文教育语料，数据量: {len(dataset)}")
            except Exception as e2:
                print(f"Fineweb加载失败: {e2}")
                try:
                    # 选项3: Beautiful Chinese语料
                    print("尝试加载Beautiful Chinese语料...")
                    dataset = load_dataset("Seikaijyu/Beautiful-Chinese", split="train[:10%]", cache_dir=args.cache_dir)
                    print(f"✓ 使用Beautiful Chinese数据集，数据量: {len(dataset)}")
                except Exception as e3:
                    print(f"Beautiful Chinese加载失败: {e3}")
                    # 选项4: 使用本地示例文件
                    if os.path.exists('data/sample_chinese.txt'):
                        print("⚠️  所有在线数据集加载失败，使用本地示例文件")
                        dataset = load_dataset('text', data_files='data/sample_chinese.txt', split='train', cache_dir=args.cache_dir)
                        print(f"✓ 加载本地文件，数据量: {len(dataset)}")
                    else:
                        print("❌ 所有中文数据集加载失败，使用英文数据集")
                        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=args.cache_dir)
    else:
        # 英文数据集
        print(f"加载数据集: {args.dataset}")
        if args.dataset == 'wikitext':
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=args.cache_dir)
        else:
            dataset = load_dataset(args.dataset, cache_dir=args.cache_dir)
    
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
        remove_columns=columns_to_remove
    )
    
    return tokenized_datasets

def main():
    args = parse_args()
    
    # 初始化 WandB
    use_wandb = args.wandb_project is not None and WANDB_AVAILABLE
    if use_wandb:
        # 配置 WandB
        wandb_config = {
            "model_size": args.model_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "fp16": args.fp16,
            "bf16": args.bf16,
            "use_chinese": args.use_chinese,
        }
        
        # 初始化 WandB
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=wandb_config,
            tags=["gpt2", args.model_size, "single-gpu"]
        )
        print(f"✓ WandB 已初始化 - 项目: {args.wandb_project}")
        if args.wandb_run_name:
            print(f"  运行名称: {args.wandb_run_name}")
    elif args.wandb_project and not WANDB_AVAILABLE:
        print("⚠️  WandB 项目已指定但 wandb 未安装")
        print("   请运行: pip install wandb")
    
    if torch.version.hip is not None:
        backend = "ROCm"
    elif torch.version.cuda is not None:
        backend = "CUDA"
    else:
        backend = "CPU"
    
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device} ({backend})")
    
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU设备: {gpu_name}")
        print(f"GPU内存: {gpu_memory:.2f} GB")
        
        # 记录到 WandB
        if use_wandb:
            wandb.config.update({
                "gpu_name": gpu_name,
                "gpu_memory_gb": gpu_memory,
                "backend": backend
            })
    
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
    model.to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params/1e6:.2f}M")
    print(f"可训练参数: {trainable_params/1e6:.2f}M")
    
    # 显示混合精度状态
    if args.fp16:
        print("✓ 使用FP16混合精度训练")
    elif args.bf16:
        print("✓ 使用BF16混合精度训练")
    else:
        print("使用FP32全精度训练")
    
    # 加载数据集
    tokenized_datasets = load_and_prepare_dataset(args, tokenizer)
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT使用因果语言模型，不是MLM
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
    
    # 训练参数
    # 根据是否使用 WandB 设置 report_to
    report_to_list = ["tensorboard"]
    if use_wandb:
        report_to_list.append("wandb")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
        logging_dir=f"{args.output_dir}/logs",
        eval_strategy="steps" if has_eval else "no",  # 只在有validation数据时启用
        eval_steps=500 if has_eval else None,
        fp16=args.fp16,  # 混合精度训练
        bf16=args.bf16,  # BFloat16混合精度
        report_to=report_to_list,  # 同时支持 TensorBoard 和 WandB
        save_safetensors=True,
        run_name=args.wandb_run_name if use_wandb else None,  # WandB 运行名称
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("\n" + "="*50)
    print("开始训练...")
    print("="*50 + "\n")
    
    trainer.train()
    
    # 保存模型
    print(f"\n保存模型到: {args.model_save_dir}")
    model.save_pretrained(args.model_save_dir)
    tokenizer.save_pretrained(args.model_save_dir)
    
    print("\n训练完成!")
    print(f"模型保存在: {args.model_save_dir}")
    print(f"日志保存在: {args.output_dir}/logs")
    print("\n使用以下命令查看训练日志:")
    print(f"  tensorboard --logdir={args.output_dir}/logs")

if __name__ == "__main__":
    main()
