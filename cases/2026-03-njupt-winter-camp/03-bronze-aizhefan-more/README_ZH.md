# AI助手项目概述

## 项目摘要

本项目由两个主要组件组成：

1. **LLM（大语言模型）** - 一个基于AMD ROCm优化的3B参数对话语言模型
2. **虚拟桌面宠物** - 一个集成了人格模拟和WSL2命令执行能力的AI桌面宠物系统

## 项目结构

```
├── llm/                # 3B参数对话语言模型
│   ├── configs/        # 配置文件
│   ├── core/           # 核心模型架构
│   ├── data/           # 数据处理
│   ├── inference/      # 推理系统
│   ├── scripts/        # 训练和推理脚本
│   └── training/       # 训练系统
├── Virtual Desktop Companion/  # AI桌面宠物系统
│   ├── client/         # 客户端代码
│   └── server/         # 服务器端代码
├── README.md           # 英文文档
├── README_ZH.md        # 中文文档
├── requirements.txt    # 依赖项
├── main.ipynb          # 英文笔记本
└── main_zh.ipynb       # 中文笔记本
```

## LLM组件

### 模型架构

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

### 硬件要求

- **GPU**: AMD Radeon 8060S (RDNA 3.5, gfx1151) 或更高
- **显存**: 64GB共享内存 (最小16GB)
- **CPU**: 4核以上
- **内存**: 16GB以上
- **ROCm**: 7.10+

## 虚拟桌面宠物组件

### 系统架构

- **客户端-服务器架构**: 分离的客户端和服务器组件
- **客户端**: 提供用户界面，处理用户输入，显示AI回复
- **服务器**: 处理客户端请求，执行LLM推理，管理知识库

### 核心功能

1. **人格模拟**: 具有独特性格和情绪状态的AI桌面宠物
2. **知识检索**: 针对WSL2的专业知识RAG系统
3. **WSL2命令执行**: 通过本地服务器执行WSL2命令
4. **多模式交互**: 针对不同场景的不同交互模式
5. **记忆系统**: 跟踪对话历史并构建用户画像

### 关键组件

- **客户端应用**: 基于Vue.js的前端，带有Live2D角色动画
- **本地服务器**: 用于WSL2命令执行的Node.js服务器
- **服务器API**: 基于FastAPI的后端，用于LLM推理
- **RAG系统**: 管理专业知识并提供检索服务
- **提示词组装器**: 构建LLM提示词并管理响应策略
- **记忆系统**: 管理会话历史并构建用户画像

## 技术亮点

1. **AMD ROCm优化**: 利用AMD GPU能力实现高效的模型训练和推理
2. **FlashAttention 2**: 优化的注意力机制，提高内存使用效率
3. **RAG技术**: 通过检索增强生成提升知识能力
4. **多API集成**: 结合不同的LLM API以获得最佳性能
5. **本地命令执行**: 通过本地服务器安全执行WSL2命令
6. **人格模拟**: 丰富的人格和情绪状态，提供引人入胜的交互

## 安装和设置

### LLM组件

```bash
# 安装依赖
pip install -r llm/requirements.txt

# 安装ROCm版本的PyTorch
pip install torch --index-url https://download.pytorch.org/whl/rocm7.1

# 训练模型
python llm/scripts/train.py --train_data ./data/train --eval_data ./data/eval

# 运行推理
python llm/scripts/inference.py --model_path ./checkpoints/final --prompt "你好"
```

### 虚拟桌面宠物

```bash
# 服务器设置
cd Virtual Desktop Companion/server
pip install -r requirements.txt
python main.py

# 客户端设置
cd Virtual Desktop Companion/client
npm install
node local-server.js
# 在浏览器中打开index.html
```

## 使用场景

1. **个人AI助手**: 与AI桌面宠物进行自然对话
2. **技术支持**: 获取WSL2命令和故障排除的帮助
3. **知识检索**: 访问关于WSL2和相关技术的信息
4. **命令执行**: 通过自然语言执行WSL2命令
5. **人格交互**: 根据上下文体验不同的交互模式

## 未来发展

1. **知识库扩展**: 覆盖更多技术领域和主题
2. **人格增强**: 深化人格模拟和情绪智能
3. **安全优化**: 提高命令执行的安全性和效率
4. **跨平台支持**: 扩展到更多操作系统和环境
5. **个性化选项**: 为用户提供更多定制功能

## 结论

本项目展示了先进的LLM技术与实际应用的集成，创建了一个综合的AI助手系统，结合了人格模拟、知识检索和命令执行能力。通过利用AMD ROCm优化，系统在提供引人入胜和有用的用户交互的同时，实现了高效的性能。其实说白了两个项目没多少关系，可以分别去学习一个主要是机器学习方向，另外一个主要是文本处理方向。
