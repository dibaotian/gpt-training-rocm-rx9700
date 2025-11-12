# GPT 训练过程详解

## 目录
1. [GPT 训练基本原理](#gpt-训练基本原理)
2. [训练代码详细解读](#训练代码详细解读)
3. [训练过程步骤](#训练过程步骤)
4. [参数详解](#参数详解)
5. [监控和调试](#监控和调试)
6. [常见问题和解决方案](#常见问题和解决方案)

---

## GPT 训练基本原理

### 什么是 GPT？

GPT (Generative Pre-trained Transformer) 是一种基于 Transformer 架构的语言模型。

**核心思想**：
- **自回归 (Autoregressive)**：根据前面的文本预测下一个词
- **预训练 (Pre-training)**：在大量文本上学习语言的统计规律
- **生成式 (Generative)**：可以生成连贯的文本

### 训练目标

GPT 的训练目标是**最大化给定前文的情况下，预测下一个词的概率**。

```
给定: "The cat sat on the"
预测: "mat" (或其他合理的词)

数学表达:
最大化 P(mat | The cat sat on the)
```

### Transformer 架构

```
输入文本 "Hello world"
    ↓
[Tokenization] 分词
    ↓
[Embedding] 词嵌入 + 位置编码
    ↓
[Transformer Block 1]
    - Multi-Head Attention (多头注意力)
    - Feed Forward (前馈网络)
    ↓
[Transformer Block 2]
    ...
    ↓
[Transformer Block N]
    ↓
[Language Model Head] 预测下一个词
    ↓
输出: 下一个词的概率分布
```

---

## 训练代码详细解读

### 完整训练代码结构

```python
# train_single_gpu.py 的完整结构

1. 导入依赖
2. 定义参数解析
3. 配置模型
4. 加载数据
5. 数据预处理
6. 创建训练器
7. 执行训练
8. 保存模型
```

### 逐步详解

#### 步骤 1: 导入依赖

```python
import torch  # PyTorch 核心库
import torch.nn as nn  # 神经网络模块
from transformers import (
    GPT2LMHeadModel,  # GPT-2 模型（包含 LM Head）
    GPT2Config,       # 模型配置
    GPT2Tokenizer,    # 分词器
    Trainer,          # 训练器（封装训练循环）
    TrainingArguments,# 训练参数
    DataCollatorForLanguageModeling  # 数据整理器
)
from datasets import load_dataset  # 加载数据集
```

**为什么用这些库？**
- `transformers`: Hugging Face 提供的预训练模型库
- `datasets`: 方便的数据集加载和处理
- `torch`: 底层深度学习框架

#### 步骤 2: 参数解析

```python
def parse_args():
    parser = argparse.ArgumentParser(description='单GPU GPT训练')
    
    # 模型配置
    parser.add_argument('--model_size', type=str, default='small')
    # small: 117M 参数，适合大多数场景
    
    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=8)
    # 批次大小：一次处理多少个样本
    # 越大训练越快，但占用更多显存
    
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    # 学习率：控制参数更新的步长
    # 太大：训练不稳定；太小：训练太慢
    
    parser.add_argument('--epochs', type=int, default=3)
    # 训练轮数：完整遍历数据集的次数
    
    return parser.parse_args()
```

#### 步骤 3: 模型配置

```python
# 定义模型架构
config = GPT2Config(
    vocab_size=50257,      # 词汇表大小（GPT-2 标准）
    n_positions=1024,      # 最大序列长度
    n_embd=768,            # 嵌入维度（向量大小）
    n_layer=12,            # Transformer 层数
    n_head=12,             # 注意力头数
)
```

**参数解释**：

1. **vocab_size (词汇表大小)**
   ```
   词汇表: {"hello": 0, "world": 1, ..., "cat": 50256}
   文本 "hello world" → [0, 1]
   ```

2. **n_positions (序列长度)**
   ```
   最多能处理 1024 个 token 的文本
   超过会被截断
   ```

3. **n_embd (嵌入维度)**
   ```
   每个词被表示为 768 维的向量
   "cat" → [0.1, -0.5, 0.3, ..., 0.8]  # 768 个数字
   ```

4. **n_layer (层数)**
   ```
   12 层 Transformer 块堆叠
   越多层，模型越深，能力越强，但也越慢
   ```

5. **n_head (注意力头数)**
   ```
   多头注意力：12 个头并行处理
   每个头关注不同的模式
   ```

**模型大小对比**：

| 配置 | 层数 | 嵌入维度 | 注意力头 | 参数量 | 显存需求 |
|------|------|---------|---------|--------|---------|
| tiny | 6 | 384 | 6 | ~50M | ~1GB |
| small | 12 | 768 | 12 | ~117M | ~3GB |
| medium | 24 | 1024 | 16 | ~345M | ~8GB |

#### 步骤 4: 创建模型

```python
model = GPT2LMHeadModel(config)
model.to(device)  # 移动到 GPU

# 模型架构
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)        # 词嵌入
    (wpe): Embedding(1024, 768)         # 位置编码
    (h): ModuleList(
      (0-11): 12 x GPT2Block(           # 12 个 Transformer 块
        (attn): GPT2Attention(...)       # 注意力层
        (mlp): GPT2MLP(...)              # 前馈网络
      )
    )
  )
  (lm_head): Linear(768, 50257)         # 语言模型头
)
```

**参数量计算**：
```
总参数 = 词嵌入 + 位置编码 + Transformer块 + LM Head
       = 50257×768 + 1024×768 + (12层参数) + 768×50257
       ≈ 117M
```

#### 步骤 5: 加载和处理数据

```python
# 加载数据集
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# dataset 结构:
# DatasetDict({
#     'train': Dataset({...}),
#     'validation': Dataset({...}),
#     'test': Dataset({...})
# })

# 数据样本:
# {
#     'text': 'The quick brown fox jumps over the lazy dog.'
# }
```

**数据预处理**：

```python
def tokenize_function(examples):
    # 输入: {'text': ['Hello world', 'GPT is cool', ...]}
    return tokenizer(
        examples['text'],
        truncation=True,        # 超过 max_length 截断
        max_length=512,         # 最大长度
        padding='max_length'    # 填充到统一长度
    )
    # 输出: {
    #     'input_ids': [[101, 2023, ...], [101, 2024, ...], ...],
    #     'attention_mask': [[1, 1, ...], [1, 1, ...], ...]
    # }

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,              # 批量处理（更快）
    remove_columns=['text']    # 删除原始文本列
)
```

**Tokenization 示例**：

```python
文本: "Hello world!"

步骤1 - 分词:
"Hello" → token_id: 15496
"world" → token_id: 995
"!" → token_id: 0

步骤2 - 添加特殊token (如果需要):
[BOS] Hello world! [EOS]
↓
[50256, 15496, 995, 0, 50256]

步骤3 - 填充到固定长度:
[50256, 15496, 995, 0, 50256, 0, 0, ..., 0]  # 填充到512
对应的 attention_mask:
[1, 1, 1, 1, 1, 0, 0, ..., 0]  # 1=真实token, 0=填充
```

#### 步骤 6: 数据整理器

```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # GPT 用自回归，不是 MLM (Masked Language Model)
)
```

**数据整理器的作用**：

```python
# 输入: 一批样本
[
    {'input_ids': [1, 2, 3, 4, 5]},
    {'input_ids': [6, 7, 8, 9]},
    ...
]

# 输出: 整理后的批次
{
    'input_ids': torch.tensor([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 0],  # 填充
        ...
    ]),
    'attention_mask': torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0],  # 标记填充位置
        ...
    ]),
    'labels': torch.tensor([
        [2, 3, 4, 5, -100],  # 标签（下一个词）
        [7, 8, 9, 0, -100],  # -100 表示忽略
        ...
    ])
}
```

**为什么需要 labels？**

GPT 训练是预测下一个词：
```
输入: "The cat sat"
标签: "cat sat on" (输入向右移一位)

计算损失:
给定 "The" 预测 "cat"
给定 "The cat" 预测 "sat"
给定 "The cat sat" 预测 "on"
```

#### 步骤 7: 训练参数

```python
training_args = TrainingArguments(
    output_dir='./output',              # 输出目录
    num_train_epochs=3,                 # 训练 3 轮
    per_device_train_batch_size=8,      # 每个 GPU 批次大小
    learning_rate=5e-5,                 # 学习率
    save_steps=1000,                    # 每 1000 步保存一次
    logging_steps=100,                  # 每 100 步记录一次
    eval_strategy="steps",              # 评估策略
    eval_steps=500,                     # 每 500 步评估一次
)
```

#### 步骤 8: 创建训练器

```python
trainer = Trainer(
    model=model,                        # 模型
    args=training_args,                 # 训练参数
    train_dataset=tokenized_datasets['train'],      # 训练集
    eval_dataset=tokenized_datasets['validation'],  # 验证集
    data_collator=data_collator,        # 数据整理器
)
```

**Trainer 做什么？**

Trainer 封装了完整的训练循环：

```python
# 伪代码
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # 前向传播
        outputs = model(batch['input_ids'])
        loss = compute_loss(outputs, batch['labels'])
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        optimizer.zero_grad()
        
        # 记录日志
        if step % logging_steps == 0:
            log_metrics(loss, learning_rate, ...)
        
        # 评估
        if step % eval_steps == 0:
            eval_loss = evaluate(model, eval_dataset)
            log_metrics(eval_loss)
        
        # 保存检查点
        if step % save_steps == 0:
            save_checkpoint(model, step)
```

#### 步骤 9: 执行训练

```python
trainer.train()
```

**训练过程中发生了什么？**

```
Epoch 1/3:
  Step 0-99:    Loss: 4.523 → 4.201 → 3.987 → ...
  Step 100:     [记录] Loss: 3.845, LR: 5e-5
  Step 500:     [评估] Val Loss: 3.756
  Step 1000:    [保存检查点]
  ...

Epoch 2/3:
  Step 1001:    Loss: 3.234 → 3.102 → ...
  ...

Epoch 3/3:
  Step 2001:    Loss: 2.891 → ...
  ...

训练完成！
```

**损失函数计算**：

```python
# 交叉熵损失 (Cross Entropy Loss)
def compute_loss(logits, labels):
    """
    logits: [batch_size, seq_len, vocab_size] 
            模型对每个位置的每个词的预测分数
    labels: [batch_size, seq_len]
            真实的下一个词
    """
    loss = -log(P(正确的词 | 上下文))
    return loss.mean()

# 示例:
给定 "The cat"，模型预测:
{
    "sat": 0.3,  ← 正确答案
    "is": 0.2,
    "runs": 0.15,
    ...
}

Loss = -log(0.3) = 1.20

如果模型预测很准确:
{
    "sat": 0.9,  ← 正确答案
    "is": 0.05,
    ...
}
Loss = -log(0.9) = 0.11  (更低，更好)
```

#### 步骤 10: 保存模型

```python
model.save_pretrained("./gpt_model")
tokenizer.save_pretrained("./gpt_model")
```

**保存了什么？**

```
./gpt_model/
├── config.json              # 模型配置
├── pytorch_model.bin        # 模型权重 (或 model.safetensors)
├── tokenizer_config.json    # 分词器配置
├── vocab.json               # 词汇表
└── merges.txt               # BPE 合并规则
```

---

## 训练过程步骤

### 完整训练流程图

```
┌─────────────────────────────────────────────────────────┐
│ 1. 初始化                                                │
│    - 加载配置                                           │
│    - 创建模型                                           │
│    - 初始化优化器                                       │
└────────────────┬────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────┐
│ 2. 数据准备                                              │
│    - 加载数据集                                         │
│    - Tokenization                                       │
│    - 创建 DataLoader                                    │
└────────────────┬────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────┐
│ 3. 训练循环 (每个 epoch)                                 │
│                                                          │
│   ┌──────────────────────────────────────────┐          │
│   │ 3.1 前向传播                              │          │
│   │     输入 → 模型 → 输出 logits            │          │
│   └──────────────┬───────────────────────────┘          │
│                  ↓                                       │
│   ┌──────────────────────────────────────────┐          │
│   │ 3.2 计算损失                              │          │
│   │     Loss = CrossEntropy(logits, labels)  │          │
│   └──────────────┬───────────────────────────┘          │
│                  ↓                                       │
│   ┌──────────────────────────────────────────┐          │
│   │ 3.3 反向传播                              │          │
│   │     Loss.backward() → 计算梯度           │          │
│   └──────────────┬───────────────────────────┘          │
│                  ↓                                       │
│   ┌──────────────────────────────────────────┐          │
│   │ 3.4 更新参数                              │          │
│   │     Optimizer.step() → 更新权重          │          │
│   └──────────────┬───────────────────────────┘          │
│                  ↓                                       │
│   ┌──────────────────────────────────────────┐          │
│   │ 3.5 记录和评估                            │          │
│   │     - 记录损失                            │          │
│   │     - 在验证集上评估                      │          │
│   │     - 保存检查点                          │          │
│   └──────────────────────────────────────────┘          │
└────────────────┬────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────┐
│ 4. 训练结束                                              │
│    - 保存最终模型                                       │
│    - 生成训练报告                                       │
└─────────────────────────────────────────────────────────┘
```

### 详细训练步骤

#### 第 1 步：初始化

```bash
# 在容器内执行
python3 train_single_gpu.py --model_size small
```

**系统输出**：
```
使用设备: cuda (ROCm)
GPU设备: AMD Radeon RX 9700
GPU内存: 32
.00 GB

模型配置: small
  - 层数: 12
  - 嵌入维度: 768
  - 注意力头: 12

创建模型...
模型参数量: 117.22M
```

#### 第 2 步：加载数据

```
加载数据集: wikitext
开始tokenize数据集...
数据集包含以下split: ['train', 'validation', 'test']
要删除的列: ['text']

Map: 100%|████████████| 36718/36718 [00:05<00:00, 7234.12 examples/s]
Map: 100%|████████████| 3760/3760 [00:00<00:00, 7156.34 examples/s]
```

#### 第 3 步：训练

```
==================================================
开始训练...
==================================================

Epoch 1/3
  0%|                                | 0/4590 [00:00<?, ?it/s]
Step 1:   Loss: 10.8245  LR: 5e-05  Samples/sec: 45.2
Step 10:  Loss: 9.2341   LR: 5e-05  Samples/sec: 48.7
Step 50:  Loss: 7.5623   LR: 5e-05  Samples/sec: 51.3
Step 100: Loss: 6.1234   LR: 5e-05  Samples/sec: 52.1
          ↑ 损失下降 = 模型在学习

Step 500: [评估] Val Loss: 5.2341
          保存检查点...

Epoch 2/3
Step 1000: Loss: 4.3456  LR: 5e-05
           ↑ 损失继续下降

Epoch 3/3
Step 2000: Loss: 3.1234  LR: 5e-05
           ↑ 模型越来越好

训练完成！总时间: 2.5 小时
```

**GPU 显存使用**：

```
Device  GPU%  VRAM%  Power   Temp
0       99%   45%    285W    68°C
        ↑     ↑      ↑       ↑
      计算  显存   功耗    温度
```

#### 第 4 步：保存模型

```
保存模型到: ./gpt_model

模型文件:
./gpt_model/
├── config.json (946 bytes)
├── pytorch_model.bin (469 MB)  ← 模型权重
├── tokenizer_config.json
└── vocab.json
```

---

## 参数详解

### 模型参数

#### vocab_size (词汇表大小)

```python
vocab_size = 50257

# 影响:
# - 模型能识别的不同词的数量
# - Embedding 层的大小
# - 输出层的大小
# - 总参数量

# GPT-2 使用 Byte-Pair Encoding (BPE)
# 50257 = 50000 个常用 token + 256 个字节 + 1 个特殊 token
```

#### n_positions (最大序列长度)

```python
n_positions = 1024

# 影响:
# - 能处理的最大文本长度
# - 位置编码的大小
# - 注意力矩阵的大小 (O(n²))

# 示例:
# 512 tokens ≈ 350-400 个英文单词
# 1024 tokens ≈ 700-800 个英文单词
```

#### n_embd (嵌入维度)

```python
n_embd = 768

# 影响:
# - 每个 token 的向量大小
# - 模型的"带宽"
# - 大部分层的宽度
# - 参数量的主要贡献者

# 更大的维度:
# ✓ 更强的表达能力
# ✗ 更多的参数
# ✗ 更慢的计算
```

#### n_layer (层数)

```python
n_layer = 12

# 影响:
# - 模型的深度
# - 抽象能力的层次

# 层次理解:
# 第 1-4 层:  学习基础语法和词汇
# 第 5-8 层:  学习语义和上下文
# 第 9-12 层: 学习复杂推理和生成
```

#### n_head (注意力头数)

```python
n_head = 12

# 影响:
# - 并行注意力的数量
# - 每个头关注不同的模式

# 示例 (假设):
# Head 1: 关注主谓关系
# Head 2: 关注定语修饰
# Head 3: 关注时态
# ...
# Head 12: 关注长距离依赖
```

### 训练参数

#### batch_size (批次大小)

```python
batch_size = 8

# 含义: 一次训练处理 8 个样本

# 影响:
# - 显存占用: batch_size ↑ → VRAM ↑
# - 训练速度: batch_size ↑ → GPU 利用率 ↑
# - 梯度估计: batch_size ↑ → 更稳定
# - 泛化能力: 太大可能过拟合

# 选择建议:
# - 显存充足: 尽量大 (32, 64)
# - 显存紧张: 较小 (8, 16) + 梯度累积
```

#### learning_rate (学习率)

```python
learning_rate = 5e-5  # 0.00005

# 含义: 参数更新的步长

# 影响:
# - 太大 (1e-3): 训练不稳定，可能发散
# - 太小 (1e-6): 训练太慢，可能卡住
# - 刚好 (5e-5): 稳定且高效

# 常用范围:
# - 从头训练: 1e-4 到 1e-3
# - 微调预训练模型: 1e-5 到 5e-5
```

#### gradient_accumulation_steps (梯度累积)

```python
gradient_accumulation_steps = 4

# 含义:
# 累积 4 个小批次的梯度，然后更新一次参数
# 有效批次大小 = batch_size × gradient_accumulation_steps

# 示例:
# batch_size = 8, accumulation = 4
# 有效批次 = 8 × 4 = 32

# 优势:
# ✓ 模拟大批次，但显存占用小
# ✓ 适合显存有限的情况
```

#### max_length (最大序列长度)

```python
max_length = 512

# 影响:
# - 能处理的文本长度
# - 显存占用 (O(n²))
# - 训练速度

# 权衡:
# 更长序列:
# ✓ 能看到更多上下文
# ✗ 显存占用大
# ✗ 训练更慢
```

### 优化器参数

```python
# Adam 优化器参数
optimizer = AdamW(
    lr=5e-5,              # 学习率
    betas=(0.9, 0.999),   # 动量参数
    eps=1e-8,             # 数值稳定性
    weight_decay=0.01,    # 权重衰减 (L2 正则化)
)

# weight_decay 作用:
# 防止过拟合，让权重不要太大
```

---

## 监控和调试

### 训练指标

#### 1. Loss (损失)

```
Epoch 1: Loss 10.5 → 8.2 → 6.7 → 5.4 → ...
Epoch 2: Loss 4.9 → 4.1 → 3.8 → ...
Epoch 3: Loss 3.5 → 3.2 → 3.0 → ...
```

**正常情况**：
- ✓ 持续下降
- ✓ 训练损失 < 验证损失（轻微）

**异常情况**：
- ✗ 不下降 → 学习率太小或数据有问题
- ✗ 震荡 → 学习率太大
- ✗ NaN → 数值不稳定，降低学习率
- ✗ 训练损失 << 验证损失 → 过拟合

#### 2. Learning Rate (学习率)

```
使用学习率调度器:

Warmup (前 10%):
  Step 0-100:   LR: 0 → 5e-5  (线性增加)

Training:
  Step 101-900: LR: 5e-5 (恒定)

Decay (后期):
  Step 901-1000: LR: 5e-5 → 1e-5 (余弦衰减)
```

#### 3. GPU 指标

```bash
# 实时监控
watch -n 1 rocm-smi

# 输出:
Device  GPU%  VRAM%  Power   SCLK     MCLK     Temp
0       99%   65%    290W    2934Mhz  1258Mhz  72°C

# 理想状态:
GPU%:  95-100%  (充分利用)
VRAM%: 60-80%   (不浪费也不OOM)
Power: 接近最大  (满载工作)
Temp:  <80°C    (安全温度)
```

### 使用 TensorBoard

```bash
# 启动 TensorBoard (容器内)
tensorboard --logdir=./output_single/logs --bind_all

# 主机浏览器访问
http://localhost:6006
```

**TensorBoard 界面**：

```
Scalars (标量):
├── train/loss          # 训练损失曲线
├── train/learning_rate # 学习率变化
├── eval/loss           # 验证损失
└── train/samples_per_second  # 训练速度

Graphs (图):
└── model_graph         # 模型结构可视化

Distributions (分布):
├── layer1/weights      # 各层权重分布
└── layer1/gradients    # 梯度分布
```

### 常见训练曲线

#### 健康的训练曲线

```
Loss
 |
 |  \
 |   \___
 |       \___
 |           \___________
 |________________________
 0     500    1000   1500
        Steps

✓ 平滑下降
✓ 训练集和验证集趋势相似
✓ 最终收敛
```

#### 过拟合

```
Loss
 |
 |  Train: \_________
 |
 |  Val:    \   /\  /\
 |           \_/  \/
 |________________________
 0     500    1000   1500

✗ 训练损失持续下降
✗ 验证损失反弹
→ 需要: 正则化、Dropout、更多数据
```

#### 欠拟合

```
Loss
 |
 |  ___________________
 |
 |  Train & Val 都很高
 |________________________
 0     500    1000   1500

✗ 损失不下降
→ 需要: 更大模型、更多训练、调整学习率
```

---

## 常见问题和解决方案

### 问题 1: Out of Memory (OOM)

```
RuntimeError: HIP out of memory
```

**原因**: 显存不足

**解决方案**：

```python
# 1. 减小 batch_size
batch_size = 4  # 从 8 降到 4

# 2. 增加梯度累积
gradient_accumulation_steps = 8  # 保持有效批次不变

# 3. 减小序列长度
max_length = 256  # 从 512 降到 256

# 4. 启用混合精度 (如果支持)
fp16 = True

# 5. 启用梯度检查点
gradient_checkpointing = True
```

### 问题 2: 训练太慢

**症状**: GPU 利用率低，训练速度慢

**解决方案**：

```python
# 1. 增大 batch_size
batch_size = 32  # 提高 GPU 利用率

# 2. 增加数据加载线程
dataloader_num_workers = 4

# 3. 启用混合精度
fp16 = True  # 加速计算

# 4. 使用优化脚本
./run_single_gpu_optimized.sh
```

### 问题 3: 损失不下降

**可能原因**：

1. **学习率太小**
   ```python
   # 尝试增大学习率
   learning_rate = 1e-4  # 从 5e-5 增加到 1e-4
   ```

2. **数据问题**
   ```python
   # 检查数据
   print(tokenized_datasets['train'][0])
   # 确保有 input_ids 和 labels
   ```

3. **模型太小**
   ```python
   # 使用更大的模型
   model_size = 'medium'  # 从 small 升级到 medium
   ```

### 问题 4: 训练不稳定

**症状**: 损失震荡或出现 NaN

**解决方案**：

```python
# 1. 降低学习率
learning_rate = 1e-5

# 2. 使用梯度裁剪
max_grad_norm = 1.0  # 默认已启用

# 3. 增加 warmup
warmup_steps = 500

# 4. 减小 batch_size
batch_size = 4
```

### 问题 5: 如何恢复训练

```python
# 从检查点继续训练
trainer.train(resume_from_checkpoint="./output/checkpoint-1000")
```

---

## 总结

### 训练流程回顾

1. **准备环境** → Docker 容器 + GPU
2. **配置模型** → 选择模型大小和参数
3. **加载数据** → 下载数据集并 tokenize
4. **创建训练器** → 配置训练参数
5. **执行训练** → 开始训练循环
6. **监控进度** → TensorBoard + rocm-smi
7. **保存模型** → 保存训练好的模型
8. **测试生成** → 验证模型效果

### 关键点

- ✅ **批次大小**: 根据显存调整，越大越好
- ✅ **学习率**: 从头训练用 1e-4，微调用 5e-5
- ✅ **梯度累积**: 显存不够时使用
- ✅ **监控**: 实时查看损失和 GPU 使用率
- ✅ **保存检查点**: 定期保存，以防中断

### 下一步

1. 运行基础训练，熟悉流程
2. 使用优化脚本提升性能
3. 尝试不同的模型大小
4. 在自己的数据上训练
5. 探索分布式多卡训练

---

**参考其他文档**：
- `TRAINING_PLAN.md` - 完整训练计划
- `GPU_TRAINING_OPTIMIZATION.md` - 性能优化指南
- `DOCKER_SETUP.md` - Docker 环境配置
