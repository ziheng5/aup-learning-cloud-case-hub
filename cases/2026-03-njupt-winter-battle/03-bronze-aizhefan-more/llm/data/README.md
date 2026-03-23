# 数据准备指南

## 数据格式

### 1. JSONL格式（推荐）
每行一个JSON对象：

```json
{"text": "这是一段训练文本。"}
{"text": "这是另一段训练文本。"}
```

### 2. 纯文本格式
段落之间用空行分隔：

```
这是第一段训练文本。

这是第二段训练文本。

这是第三段训练文本。
```

### 3. 对话格式
```json
{"conversations": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]}
```

---

## 快速开始

### 方式一：使用示例数据（测试）

```bash
# 复制示例数据到训练目录
mkdir -p data/train data/eval
cp data/example_data.jsonl data/train/
cp data/example_data.jsonl data/eval/
```

### 方式二：准备自己的数据

#### 步骤1：收集数据

推荐的数据来源：

| 数据源 | 网址 | 说明 |
|--------|------|------|
| Wikipedia | https://dumps.wikimedia.org/ | 中文维基百科 |
| OSCAR | https://oscar-project.org/ | 大规模多语言语料 |
| Common Crawl | https://commoncrawl.org/ | 网页爬虫数据 |

#### 步骤2：清洗数据

```bash
# 清洗JSONL格式数据
python data/preprocess.py clean \
    --input raw_data/wiki.jsonl \
    --output data/train/wiki_clean.jsonl \
    --format jsonl \
    --min_length 10 \
    --max_length 100000

# 清洗纯文本数据
python data/preprocess.py clean \
    --input raw_data/books.txt \
    --output data/train/books_clean.jsonl \
    --format text

# 清洗对话数据
python data/preprocess.py clean \
    --input raw_data/conversations.jsonl \
    --output data/train/conversations_clean.jsonl \
    --format conversation
```

#### 步骤3：分割训练集和验证集

```bash
python data/preprocess.py split \
    --input data/all_data.jsonl \
    --train data/train/train.jsonl \
    --eval data/eval/eval.jsonl \
    --eval_ratio 0.01
```

#### 步骤4：合并多个文件

```bash
python data/preprocess.py merge \
    --input_dir data/train \
    --output data/train/merged.jsonl
```

---

## 数据量建议

| 模型规模 | 推荐数据量 | 训练步数 |
|---------|-----------|---------|
| 3B参数 | 10-100GB | 100K-500K |
| 7B参数 | 50-200GB | 200K-1M |

---

## 数据质量检查

```python
# 检查数据格式
import json

with open('data/train/train.jsonl', 'r') as f:
    for i, line in enumerate(f):
        try:
            data = json.loads(line)
            assert 'text' in data
        except Exception as e:
            print(f"Line {i}: {e}")
        if i >= 10:
            break

print("数据格式检查通过！")
```

---

## 目录结构

```
language_model/
└── data/
    ├── train/                    # 训练数据
    │   ├── wiki.jsonl
    │   ├── books.jsonl
    │   └── conversations.jsonl
    ├── eval/                     # 验证数据
    │   └── eval.jsonl
    ├── preprocess.py             # 预处理工具
    ├── example_data.jsonl        # 示例数据
    └── README.md                 # 本文件
```

---

## 常见问题

### Q: 数据太少怎么办？
A: 可以使用公开的中文语料库，或者使用数据增强技术。

### Q: 数据太大内存不够？
A: 使用流式数据集（StreamingTextDataset），不需要全部加载到内存。

### Q: 如何创建自己的分词器？
A: 可以使用SentencePiece或HuggingFace Tokenizers库训练自定义分词器。

### Q: 数据需要做哪些预处理？
A: 至少需要：去重、过滤过短/过长文本、移除特殊字符。
