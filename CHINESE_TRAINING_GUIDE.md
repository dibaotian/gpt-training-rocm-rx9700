# 中文GPT模型训练指南

## 🎉 恭喜！

您已经成功训练GPT-2 tiny模型，现在可以进行更大规模的中文训练。

## 📊 可训练的模型规模

根据您的R9700 (34GB显存)，可以训练：

| 模型大小 | 参数量 | 显存需求(估算) | 推荐批次大小 | 训练时间 |
|---------|--------|---------------|-------------|---------|
| ✅ tiny | 50M | ~2GB | 8-16 | 快速 |
| ✅ small | 117M | ~3-4GB | 8-12 | 中等 |
| ✅ medium | 345M | ~8-10GB | 4-8 | 较长 |
| ✅ large | 774M | ~16-20GB | 2-4 | 长 |
| ⚠️ xl | 1.5B | ~30GB+ | 1-2 | 很长 |

**推荐起步**：GPT-2 Small (117M参数)

## 🚀 快速开始：训练中文GPT-2 Small

### 方式一：使用训练脚本（最简单）

```bash
# 训练中文GPT-2 Small模型
python3 train_single_gpu.py \
    --model_size small \
    --use_chinese \
    --epochs 5 \
    --batch_size 8 \
    --max_length 512 \
    --output_dir ./output_chinese_small \
    --model_save_dir ./gpt_model_chinese_small

# 训练完成后测试
python3 test_generation.py \
    --model_path ./gpt_model_chinese_small \
    --prompt "从前有一座山，" \
    --max_length 100 \
    --num_return_sequences 3
```

### 方式二：使用启动脚本

```bash
# 修改run_single_gpu.sh或直接运行
./run_single_gpu.sh small 5 8
```

然后在脚本运行时添加`--use_chinese`参数。

## 📚 中文数据集选项

训练脚本已更新，会自动尝试多个中文数据集：

### 自动选择数据集（推荐）

脚本会按顺序尝试：
1. **CLUE C3数据集**（中文阅读理解）
2. **中文维基百科**（20220301.zh）
3. **备选：英文数据集**（如果中文都失败）

只需添加`--use_chinese`参数即可。

### 方案A：使用自定义中文文本文件（推荐）

如果Hugging Face数据集无法访问，使用自己的文本数据最可靠：

#### 1. 准备文本文件

```bash
# 创建数据目录
mkdir -p data

# 准备中文文本文件（每行一段文本或文章）
# 例如 data/chinese_corpus.txt
```

示例内容（`data/chinese_corpus.txt`）：
```
人工智能是计算机科学的一个分支，它企图了解智能的实质。
深度学习是机器学习的一个子领域，使用多层神经网络。
自然语言处理是人工智能的重要应用领域之一。
...
```

#### 2. 修改训练脚本使用本地数据

```python
# 在load_and_prepare_dataset函数中
if args.use_chinese:
    # 使用本地文本文件
    dataset = load_dataset('text', data_files='data/chinese_corpus.txt', split='train')
```

#### 3. 训练

```bash
python3 train_single_gpu.py \
    --model_size small \
    --use_chinese \
    --epochs 5 \
    --batch_size 16 \
    --fp16
```

### 方案B：手动下载数据集

如果网络问题导致数据集无法访问：

```bash
# 1. 设置镜像（如果在国内）
export HF_ENDPOINT=https://hf-mirror.com

# 2. 手动下载中文维基百科
python3 << 'EOF'
from datasets import load_dataset

# 下载并缓存
dataset = load_dataset("wikipedia", "20220301.zh", split="train[:10%]")
print(f"数据集大小: {len(dataset)}")
EOF
```

### 方案C：使用其他可访问的中文数据集

```python
# 一些可能可用的数据集：

# 1. CLUECorpusSmall
dataset = load_dataset("clue", "cluecorpussmall", split="train")

# 2. 中文新闻数据
dataset = load_dataset("thu-coai/CDial-GPT", split="train[:10%]")

# 3. 中文问答数据
dataset = load_dataset("shibing624/nli_zh", split="train")
```

## 🔧 训练不同规模模型的命令

### GPT-2 Small (推荐开始)

```bash
python3 train_single_gpu.py \
    --model_size small \
    --use_chinese \
    --epochs 5 \
    --batch_size 8 \
    --max_length 512 \
    --learning_rate 5e-5 \
    --output_dir ./output_chinese_small \
    --model_save_dir ./gpt_model_chinese_small
```

**显存使用**：~4-5GB
**训练时间**：根据数据量，可能数小时

### GPT-2 Medium（需要更多显存）

```bash
python3 train_single_gpu.py \
    --model_size medium \
    --use_chinese \
    --epochs 3 \
    --batch_size 4 \
    --max_length 512 \
    --learning_rate 3e-5 \
    --output_dir ./output_chinese_medium \
    --model_save_dir ./gpt_model_chinese_medium
```

**显存使用**：~10-12GB
**训练时间**：较长

### 更大模型（需要修改代码）

如果要训练GPT-2 Large (774M)，需要修改`train_single_gpu.py`：

```python
# 在get_model_config函数中添加：
'large': {
    'n_positions': 1024,
    'n_embd': 1280,
    'n_layer': 36,
    'n_head': 20,
}
```

然后：
```bash
python3 train_single_gpu.py \
    --model_size large \
    --use_chinese \
    --epochs 3 \
    --batch_size 2 \
    --max_length 512 \
    --gradient_accumulation_steps 4 \
    --output_dir ./output_chinese_large \
    --model_save_dir ./gpt_model_chinese_large
```

## 💡 显存优化技巧

如果遇到OOM（显存不足）错误：

### 技巧1：减小批次大小
```bash
--batch_size 4  # 或 2
```

### 技巧2：减小序列长度
```bash
--max_length 256  # 或 128
```

### 技巧3：使用梯度累积
```bash
--gradient_accumulation_steps 4  # 累积4步再更新
```

这样可以模拟更大的批次：
- 实际批次：2
- 累积步数：4  
- 有效批次：2 × 4 = 8

### 技巧4：混合精度训练（如果ROCm支持）
```bash
--fp16  # 或 --bf16
```

## 🎯 推荐的训练流程

### 步骤1：从Small开始（验证）

```bash
# 使用小数据集快速验证
python3 train_single_gpu.py \
    --model_size small \
    --use_chinese \
    --epochs 1 \
    --batch_size 8 \
    --max_length 256 \
    --output_dir ./output_test \
    --model_save_dir ./gpt_model_test

# 测试生成
python3 test_generation.py \
    --model_path ./gpt_model_test \
    --prompt "今天天气" \
    --max_length 50
```

### 步骤2：完整训练Small

```bash
# 使用更多数据和更多轮次
python3 train_single_gpu.py \
    --model_size small \
    --use_chinese \
    --epochs 5 \
    --batch_size 8 \
    --max_length 512 \
    --output_dir ./output_chinese_small \
    --model_save_dir ./gpt_model_chinese_small
```

### 步骤3：扩展到Medium（可选）

```bash
python3 train_single_gpu.py \
    --model_size medium \
    --use_chinese \
    --epochs 3 \
    --batch_size 4 \
    --max_length 512 \
    --output_dir ./output_chinese_medium \
    --model_save_dir ./gpt_model_chinese_medium
```

## 📝 监控训练

### 查看GPU使用
```bash
# 在另一个终端
watch -n 1 rocm-smi
```

### 查看训练日志
```bash
# 使用TensorBoard
tensorboard --logdir=./output_chinese_small/logs

# 浏览器访问
http://localhost:6006
```

### 实时查看loss
```bash
# 查看最新日志
tail -f output_chinese_small/logs/events.out.tfevents.*
```

## 🧪 测试生成效果

### 基础测试
```bash
python3 test_generation.py \
    --model_path ./gpt_model_chinese_small \
    --prompt "人工智能的未来" \
    --max_length 100 \
    --num_return_sequences 3
```

### 不同风格的提示词

```bash
# 故事生成
python3 test_generation.py \
    --model_path ./gpt_model_chinese_small \
    --prompt "从前有一个勇敢的少年，" \
    --temperature 0.9 \
    --max_length 200

# 技术文本
python3 test_generation.py \
    --model_path ./gpt_model_chinese_small \
    --prompt "深度学习是一种" \
    --temperature 0.7 \
    --max_length 150

# 诗歌风格
python3 test_generation.py \
    --model_path ./gpt_model_chinese_small \
    --prompt "春江潮水连海平，" \
    --temperature 1.0 \
    --max_length 100
```

## 📈 数据集大小建议

根据模型大小和可用时间：

| 模型 | 推荐数据量 | 训练时间估算 | 命令示例 |
|------|-----------|------------|---------|
| Small | 5-10% | 2-4小时 | `split="train[:10%]"` |
| Medium | 10-20% | 4-8小时 | `split="train[:20%]"` |
| Large | 20-50% | 8-24小时 | `split="train[:50%]"` |

修改数据集大小：
```python
# 在train_single_gpu.py中修改
dataset = load_dataset("shibing624/chinese-c4-corpus", split="train[:10%]")
```

## 🎓 高级配置

### 使用多个中文数据集

创建自定义训练脚本，混合多个数据集：

```python
from datasets import load_dataset, concatenate_datasets

# 加载多个数据集
dataset1 = load_dataset("shibing624/chinese-c4-corpus", split="train[:5%]")
dataset2 = load_dataset("clue/clue_corpus2020", split="train[:5%]")

# 合并
combined = concatenate_datasets([dataset1, dataset2])
```

### 调整学习率和优化器

```bash
# 更大的模型通常需要更小的学习率
--learning_rate 3e-5  # Small模型
--learning_rate 2e-5  # Medium模型
--learning_rate 1e-5  # Large模型
```

## ⚠️ 注意事项

1. **中文tokenizer**：
   - GPT-2原始tokenizer对中文支持较差
   - 每个中文字可能被分成多个token
   - 考虑使用专门的中文tokenizer（需要修改代码）

2. **显存监控**：
   - 密切关注GPU显存使用
   - 如果接近34GB，降低batch_size

3. **检查点保存**：
   - 训练会自动保存检查点
   - 位置：`output_*/checkpoint-XXX/`

4. **数据集下载**：
   - 中文数据集较大，首次下载需要时间
   - 使用镜像加速：`export HF_ENDPOINT=https://hf-mirror.com`

## 🎯 完整示例：训练中文GPT-2 Small

```bash
# 1. 进入Docker容器（如果使用Docker）
./docker_run.sh

# 2. 设置Hugging Face镜像（可选，加速下载）
export HF_ENDPOINT=https://hf-mirror.com

# 3. 安装依赖（首次）
pip3 install -r requirements.txt

# 4. 开始训练
python3 train_single_gpu.py \
    --model_size small \
    --use_chinese \
    --epochs 5 \
    --batch_size 8 \
    --max_length 512 \
    --learning_rate 5e-5 \
    --output_dir ./output_chinese_small \
    --model_save_dir ./gpt_model_chinese_small

# 5. 监控训练（新终端）
tensorboard --logdir=./output_chinese_small/logs

# 6. 训练完成后测试
python3 test_generation.py \
    --model_path ./gpt_model_chinese_small \
    --prompt "人工智能" \
    --max_length 100 \
    --temperature 0.8
```

## 📖 预期结果

训练完成后，您将拥有：
- ✅ 中文GPT-2模型（117M参数）
- ✅ 能够生成连贯的中文文本
- ✅ 可用于微调特定任务
- ✅ 训练日志和检查点

祝训练顺利！🎉
