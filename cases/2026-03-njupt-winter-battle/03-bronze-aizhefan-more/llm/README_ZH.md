# 南邮寒假大作战 — AMD ROCm

[English](./README.md)

## 活动信息

| 字段 | 详情 |
|------|------|
| **活动名称** | 南京邮电大学寒假大作战 — AMD ROCm |
| **赛题主题** | 基于 ROCm 的大语言模型应用开发 |
| **提交截止** | 2026 年 3 月 19 日 |
| **现场答辩** | 2026 年 3 月 22 日 |
| **地点** | 南京邮电大学 |
| **平台** | aup-learning-cloud — AMD GPU 集群（JupyterHub） |

## 背景介绍

AMD ROCm 是开源 GPU 计算平台，支持在 AMD 硬件上运行高性能 AI 任务。本次比赛要求学生在基于 ROCm 的集群环境中构建可交互的 LLM 智能应用。

**参考资料：**
- ROCm 官方文档：https://rocm.docs.amd.com/
- ROCm GitHub：https://github.com/ROCm/ROCm
- AMD CES 2025 发布：https://www.amd.com/zh-cn/newsroom/press-releases/2025-1-6-amd-announces-expanded-consumer-and-commercial-ai-.html
- AMD CES 2026 发布：https://www.amd.com/zh-cn/newsroom/press-releases/2026-1-5-amd-expands-ai-leadership-across-client-graphics-.html

## 赛题：大语言模型应用开发

**目标：** 构建一个可交互的智能应用系统（如问答助手、聊天机器人），基于集群 ROCm 环境运行。

### 开发环境

| 组件 | 详情 |
|------|------|
| **平台** | 远程 JupyterHub（Python 3.12+），通过 aup-learning-cloud 访问 |
| **Ollama API 地址** | `open-webui-ollama.open-webui:11434` |
| **可用模型** | `qwen3-coder:30b`、`gpt-oss:20b` |
| **上下文窗口** | 32K tokens |
| **推荐交互方式** | `ipywidgets`（在 Notebook 内原生交互） |
| **API 文档** | https://ollama.readthedocs.io/api/ |

### 推荐主题

#### 主题 1：领域知识问答助手（RAG 方向）
构建面向特定领域（如校规、专业课程、编程文档）的问答系统。
- **基础：** 多轮对话、记忆机制、异步/流式 API、中英文支持
- **进阶：** 完整 RAG 链路（文档解析 → 向量检索 → 增强生成）、来源标注

#### 主题 2：文本智能分析与报告助手（数据处理方向）
利用大模型对文本进行结构化提取与深度分析。
- **基础：** 摘要、情感分析、关键词提取；格式化 Markdown 报告输出
- **进阶：** 批量处理流水线（多线程/异步）、多文本对比分析报告

#### 主题 3：代码辅助编程专家（工程应用方向）
构建面向初学者的代码解释与调试助手。
- **基础：** 代码逐段说明、常见错误分析，支持 Python（Java/C++ 可选）
- **进阶：** 自动生成项目脚手架和测试用例、代码风格检查与重构建议

#### 主题 4：交互式叙事与逻辑分析（创意与逻辑方向）
基于对话引导使用 LLM 生成故事续写，并监控故事逻辑一致性。
- **基础：** 多轮续写、故事类型选择（科幻/悬疑/奇幻）、基本结构检查
- **进阶：** 一致性检测（角色/场景状态表）、三幕式结构打分

### 技术要求

- **Prompt Engineering：** 必须在文档中展示提示词迭代过程（约束、Few-shot、思维链）
- **集群资源利用：** 必须通过代码展示对集群内网 Ollama API 的调用
- **错误处理：** 必须处理 API 超时、空输入，以及上下文超长（32K）问题，至少实现以下一种方案：
  - 分段滚动处理（Chunking & Sliding Window）
  - 级联摘要生成（Recursive Summarization）
  - 动态截断与提示词压缩（Truncation & Prompt Compression）

## 奖项设置

| 奖项 | 名额 | 奖品 |
|------|------|------|
| 一等奖 | 1 队 | PYNQ Z2 开发板 × 1 + AMD 定制 T恤 × 2 + AMD 定制折叠背包 × 1 |
| 二等奖 | 2 队 | Spartan Edge FPGA 开发板 × 1 + AMD 定制帽子 × 2 + AMD 定制折叠背包 × 1 |
| 三等奖 | 4 队 | AMD 定制马克杯 × 2 + AMD 定制折叠背包 × 1 |
| 优秀奖 | 10 队 | AMD 定制折叠背包 × 2 |
| 参与奖 | 全体参赛者 | 《可定制计算》书籍 × 1（先到先得） |

## 学习资料

- DataWhale：https://www.datawhale.cn/
- DataWhale GitHub：https://github.com/datawhalechina
- AMD ModelScope 社区：https://modelscope.cn/brand/view/AMDCommunity

## 参赛作品

> 学生通过 Pull Request 提交作品，详见 [CONTRIBUTING_ZH.md](../../CONTRIBUTING_ZH.md)。

| 文件夹 | 团队 | 项目 |
|--------|------|------|
| 一共有三个都可以看一看写的不好还请见谅 | — |  |

## 如何体验所有案例

1. 打开 aup-learning-cloud → 选择 **Basic GPU Environment**
2. Git URL 填入：`https://github.com/amdjiahangpan/aup-learning-cloud-case-hub`
3. 启动后进入 `cases/2026-03-njupt-winter-battle/`
4. 打开任意作品文件夹，运行 `main.ipynb`







如果想要按照常规来运行这个成功的概率不大因为这个 1.是一个基于 ROCm 的集群环境，需要在集群内运行。2.这个环境在我所做测试的时候并不是一帆风顺，可能会有一些问题。但是等到真正教学的那一天不一定和我的环境一模一样3.我的代码在我当时做测试的时候肯定没有一点问题，但是我的没有问题是在基于我可以解决我的问题的能力的条件下的4.如果有错误是很正常的事情这时候需要自己加把油了。5.这个又不是docker所以不要想着难有简便的方式.能力都是从错误中锻炼出来的 6.如果觉得我写的还行可以来看一下我的GitHub主页我后续有成功的作品我也会提交到这里 8. AMD的GPU对于训练一个模型的环境真的很友好 9. 这个非常大的显存要学会使用自己想想怎么样加速我的这个加速方式比较老套而且也比较容易出错我不清楚下面的环境中会报什么样的错误 

## 贡献者
绝大多数都是AI写的还有少部分是我写的如果你愿意看的话那么贡献者也可以把你加上 

# 3B参数对话语言模型

面向日常对话的Decoder-only Transformer语言模型，基于AMD ROCm优化。

## 模型架构

- **模型规模**: 3B参数
- **架构**: Decoder-only Transformer (类GPT)
- **隐藏维度**: 3072
- **层数**: 32
- **注意力头**: 24 (GQA: 8个KV头)
- **中间层维度**: 8192
- **最大序列长度**: 4096
- **词表大小**: 65536

### 关键技术

1. **FlashAttention 2**: 高效注意力计算，节省显存
2. **RoPE位置编码**: 旋转位置编码，支持长序列
3. **SwiGLU激活**: 门控线性单元，性能优于GELU
4. **RMSNorm**: 高效归一化
5. **GQA分组查询注意力**: 平衡MQA和MHA的性能与质量

## 硬件要求

### 推荐配置
- **GPU**: AMD Radeon 8060S (RDNA 3.5, gfx1151) 或更高
- **显存**: 64GB共享内存 (最小16GB)
- **CPU**: 4核以上
- **内存**: 16GB以上
- **ROCm**: 7.10+

### 多服务器并行
支持3服务器数据并行训练，可扩展到7B模型。

## 安装

```bash
# 克隆项目
cd language_model

# 安装依赖
pip install -r requirements.txt

# 安装ROCm版本的PyTorch
pip install torch --index-url https://download.pytorch.org/whl/rocm7.1
```

## 快速开始

### 1. 准备数据

数据格式支持:
- JSONL: `{"text": "..."}`
- TXT: 纯文本文件

```
data/
├── train/
│   ├── data1.jsonl
│   └── data2.txt
└── eval/
    └── eval.jsonl
```

### 2. 训练

```bash
# 单GPU训练
python scripts/train.py \
    --train_data ./data/train \
    --eval_data ./data/eval \
    --tokenizer bert-base-chinese \
    --output_dir ./checkpoints \
    --batch_size 16 \
    --gradient_accumulation 4 \
    --learning_rate 2e-4 \
    --max_steps 100000 \
    --warmup_steps 2000 \
    --mixed_precision bf16 \
    --gradient_checkpointing

# 分布式训练 (3服务器)
torchrun --nproc_per_node=1 --nnodes=3 \
    scripts/train.py \
    --train_data ./data/train \
    --output_dir ./checkpoints \
    --batch_size 12
```

### 3. 推理

```bash
# 单次推理
python scripts/inference.py \
    --model_path ./checkpoints/final \
    --prompt "你好，请介绍一下自己" \
    --max_new_tokens 512 \
    --temperature 0.7

# 交互式对话
python scripts/inference.py \
    --model_path ./checkpoints/final \
    --interactive
```

## 项目结构

```
language_model/
├── configs/              # 配置文件
│   └── model_config.py   # 模型和训练配置
├── core/                 # 核心模型
│   ├── attention/        # 注意力机制 (FlashAttention + RoPE)
│   ├── embeddings/       # 嵌入层
│   ├── blocks/           # Transformer块
│   ├── feedforward/      # 前馈网络 (SwiGLU)
│   └── model.py          # 主模型类
├── data/                 # 数据处理
│   ├── dataset.py        # 数据集类
│   └── dataloader.py     # 数据加载器
├── training/             # 训练系统
│   ├── trainer.py        # 训练器
│   ├── optimizer.py      # 优化器
│   └── scheduler.py      # 学习率调度
├── inference/            # 推理系统
│   └── generator.py      # 生成器 (采样/束搜索)
├── utils/                # 工具函数
│   ├── amd_optimization.py # AMD优化
│   └── checkpoint.py     # 检查点管理
├── scripts/              # 脚本
│   ├── train.py          # 训练脚本
│   └── inference.py      # 推理脚本
├── requirements.txt      # 依赖
└── README.md             # 项目说明
```

## 显存估算

| 精度 | 模型显存 | 激活值 | KV缓存 | 优化器 | 总计 |
|------|---------|--------|--------|--------|------|
| BF16 | ~6GB | ~24GB | ~4GB | ~48GB | ~82GB |
| FP16 | ~6GB | ~24GB | ~4GB | ~48GB | ~82GB |
| FP32 | ~12GB | ~48GB | ~8GB | ~96GB | ~164GB |

## AMD优化

已启用的优化:
- FlashAttention 2
- BF16混合精度
- torch.compile
- 梯度检查点
- 梯度累积
- ROCm环境变量优化

## 引用

```bibtex
@software{llm-3b-chinese,
  title = {3B参数中文对话语言模型},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/your-repo}
}
```

## 许可证

MIT License
