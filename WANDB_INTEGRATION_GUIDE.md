# Weights & Biases (WandB) 集成指南

## 🎯 概述

已为 `train_single_gpu.py` 添加 Weights & Biases 支持，可以在云端追踪和可视化训练过程。

## 📦 安装 WandB

### 在宿主机上
```bash
pip install wandb
```

### 在 Docker 容器内
```bash
pip install wandb
```

### 在 Dockerfile 中预装
已更新 Dockerfile，重新构建镜像即可包含 wandb：
```bash
./build_docker_image.sh
```

## 🔑 设置 WandB

### 1. 注册账号
访问 https://wandb.ai 注册账号（免费）

### 2. 登录
```bash
wandb login
```

会提示输入 API key，可从以下位置获取：
https://wandb.ai/authorize

### 3. 配置（可选）
```bash
# 设置默认项目
export WANDB_PROJECT=gpt-training

# 设置团队名
export WANDB_ENTITY=your-team
```

## 🚀 使用方法

### 基础用法

```bash
python3 train_single_gpu.py \
    --model_size tiny \
    --use_chinese \
    --epochs 5 \
    --wandb_project gpt-training
```

### 完整用法

```bash
python3 train_single_gpu.py \
    --model_size tiny \
    --use_chinese \
    --epochs 5 \
    --batch_size 16 \
    --wandb_project gpt-training \
    --wandb_run_name "tiny-chinese-exp1" \
    --wandb_entity your-username
```

### 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| --wandb_project | WandB 项目名称 | gpt-training |
| --wandb_run_name | 本次运行的名称 | tiny-chinese-exp1 |
| --wandb_entity | WandB 用户名/团队名 | your-username |

## 📊 WandB 功能

### 自动追踪的指标

WandB 会自动记录：

1. **训练损失**
   - 每 100 步记录一次
   - 实时曲线图

2. **学习率**
   - 学习率调度变化
   - Warmup 过程

3. **系统指标**
   - GPU 利用率
   - GPU 内存使用
   - CPU 使用率

4. **训练配置**
   - 模型大小
   - 批次大小
   - 学习率
   - 所有超参数

### 可视化示例

WandB 仪表板会显示：

```
📈 Loss 曲线
📊 GPU 利用率图表
⚙️ 超参数表格
📝 运行日志
🔄 实时更新
```

## 🎯 实际使用示例

### 示例 1: 单次训练

```bash
python3 train_single_gpu.py \
    --model_size tiny \
    --use_chinese \
    --epochs 5 \
    --batch_size 16 \
    --bf16 \
    --wandb_project my-gpt-project \
    --wandb_run_name "baseline-run"
```

**WandB 面板：** https://wandb.ai/your-username/my-gpt-project

### 示例 2: 对比实验

```bash
# 实验 1: 小批次
python3 train_single_gpu.py \
    --model_size tiny \
    --batch_size 8 \
    --wandb_project gpt-experiments \
    --wandb_run_name "batch-8"

# 实验 2: 大批次
python3 train_single_gpu.py \
    --model_size tiny \
    --batch_size 32 \
    --wandb_project gpt-experiments \
    --wandb_run_name "batch-32"
```

WandB 会自动生成对比图表。

### 示例 3: 在 Docker 中使用

```bash
# 启动容器时挂载 wandb 配置
docker run -it --rm \
    -v ~/.netrc:/root/.netrc \
    -v ~/.config/wandb:/root/.config/wandb \
    ... \
    gpt-train-rocm:latest

# 或在容器内登录
wandb login <your-api-key>

# 然后运行训练
python3 train_single_gpu.py --wandb_project gpt-docker
```

## 🔧 高级功能

### 1. 记录自定义指标

如需在训练脚本中记录额外指标，可以添加：

```python
if use_wandb:
    wandb.log({
        "custom_metric": value,
        "step": step
    })
```

### 2. 记录模型

```python
if use_wandb:
    # 保存模型到 WandB
    wandb.save(f"{args.model_save_dir}/*")
```

### 3. 记录示例输出

```python
if use_wandb:
    # 记录生成文本示例
    wandb.log({
        "generated_text": generated_sample
    })
```

### 4. 使用 Sweeps（超参数搜索）

创建 `sweep_config.yaml`：

```yaml
program: train_single_gpu.py
method: grid
parameters:
  batch_size:
    values: [8, 16, 32]
  learning_rate:
    values: [1e-5, 5e-5, 1e-4]
  model_size:
    value: tiny
  use_chinese:
    value: true
  epochs:
    value: 3
  wandb_project:
    value: gpt-sweep
```

运行 sweep：
```bash
wandb sweep sweep_config.yaml
wandb agent <sweep-id>
```

## 💡 最佳实践

### 1. 命名规范

使用描述性的运行名称：

```bash
--wandb_run_name "tiny-chinese-lr5e5-batch16-bf16"
```

包含关键信息：
- 模型大小
- 数据集
- 关键超参数
- 特殊配置

### 2. 组织项目

```bash
# 按任务分项目
--wandb_project "gpt-pretraining"
--wandb_project "gpt-finetuning"

# 按模型分项目
--wandb_project "gpt-tiny-experiments"
--wandb_project "gpt-small-experiments"
```

### 3. 使用标签

WandB 会自动添加标签：
- 模型大小（tiny, small 等）
- single-gpu

您也可以在初始化时添加自定义标签。

### 4. 团队协作

```bash
# 使用团队空间
--wandb_entity your-team-name
```

## 🔍 故障排查

### 问题 1: wandb 未安装

```bash
⚠️  wandb未安装，如需使用请运行: pip install wandb
```

**解决：**
```bash
pip install wandb
```

### 问题 2: 未登录

```bash
wandb: ERROR Unable to authenticate
```

**解决：**
```bash
wandb login
# 输入 API key
```

### 问题 3: 网络问题

如果无法连接 WandB 服务器：

```bash
# 离线模式
export WANDB_MODE=offline

# 训练完成后同步
wandb sync output_single/wandb/latest-run
```

### 问题 4: Docker 容器中使用

**方法 A: 挂载配置**
```bash
docker run -v ~/.netrc:/root/.netrc \
           -v ~/.config/wandb:/root/.config/wandb \
           ...
```

**方法 B: 容器内登录**
```bash
# 进入容器后
wandb login <your-api-key>
```

**方法 C: 使用环境变量**
```bash
docker run -e WANDB_API_KEY=<your-api-key> ...
```

## 📈 WandB vs TensorBoard

### 功能对比

| 功能 | TensorBoard | WandB |
|------|------------|-------|
| 本地可视化 | ✅ | ✅ |
| 云端存储 | ❌ | ✅ |
| 多实验对比 | 基础 | ✅ 强大 |
| 团队协作 | ❌ | ✅ |
| 超参数搜索 | ❌ | ✅ |
| 模型版本管理 | ❌ | ✅ |
| 报告生成 | ❌ | ✅ |

### 同时使用（推荐）

脚本已配置为同时使用两者：

```bash
python3 train_single_gpu.py \
    --wandb_project gpt-training \
    ...
```

会同时输出到：
- TensorBoard: `./output_single/logs`
- WandB: 云端仪表板

## 🎯 完整示例

### 训练并追踪到 WandB

```bash
# 1. 安装并登录 WandB（一次性）
pip install wandb
wandb login

# 2. 运行训练
python3 train_single_gpu.py \
    --model_size tiny \
    --use_chinese \
    --epochs 5 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --bf16 \
    --wandb_project "gpt-chinese-training" \
    --wandb_run_name "tiny-baseline-$(date +%Y%m%d)"

# 3. 查看结果
# 终端会显示 WandB URL
# 或访问: https://wandb.ai/<your-username>/gpt-chinese-training
```

### 在 Docker 中使用

```bash
# 1. 在宿主机登录（一次性）
wandb login

# 2. 启动容器时挂载配置
docker run -it --rm \
    -v ~/.config/wandb:/root/.config/wandb \
    -v $(pwd):/workspace \
    ... \
    gpt-train-rocm:latest

# 3. 在容器内训练
python3 train_single_gpu.py \
    --wandb_project gpt-docker \
    ...
```

## ✅ 检查清单

使用 WandB 前确认：

- [ ] 已注册 WandB 账号
- [ ] 已安装 wandb: `pip install wandb`
- [ ] 已登录: `wandb login`
- [ ] （Docker）已挂载 wandb 配置或在容器内登录
- [ ] 指定了项目名称: `--wandb_project`

## 🎉 总结

### 使用 WandB 的好处

1. ✅ **云端可视化** - 任何地方都能查看训练进度
2. ✅ **自动保存** - 所有指标永久保存
3. ✅ **对比分析** - 轻松对比不同实验
4. ✅ **团队协作** - 与团队分享实验结果
5. ✅ **报告生成** - 自动生成训练报告

### 快速开始

```bash
# 安装和登录（一次性）
pip install wandb
wandb login

# 开始训练
python3 train_single_gpu.py \
    --model_size tiny \
    --wandb_project my-gpt-project
```

就这么简单！🚀
