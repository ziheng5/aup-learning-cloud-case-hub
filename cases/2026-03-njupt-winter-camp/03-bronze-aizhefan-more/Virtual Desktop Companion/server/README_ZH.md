# AI 桌面宠物服务端

## 📋 项目概述

AI 桌面宠物服务端是一个基于 FastAPI 构建的后端服务器，为 AI 桌面宠物提供智能响应、情感管理和专业知识支持。

### 核心功能
- ✅ 自然语言聊天
- ✅ 情感状态管理
- ✅ Live2D 动画控制
- ✅ 记忆存储和检索
- ✅ 专业知识整合 (RAG)
- ✅ WSL2 命令执行
- ✅ 人格模拟
- ✅ 上下文理解
- ✅ 意图识别

## 🛠️ 技术栈

- **后端框架**: FastAPI
- **LLM 集成**: Ollama API + DeepSeek API
- **记忆系统**: ChromaDB (向量数据库)
- **缓存系统**: Redis (可选)
- **部署**: Uvicorn
- **语言**: Python 3.8+

## 📁 项目结构

```
server/
├── data/                   # 数据目录
│   ├── vector_db/          # 向量数据库
│   ├── memory.db           # 深度记忆数据库
│   └── reflection_logs/    # 反思日志
├── data_collection/        # 数据收集脚本
│   ├── processed_data/     # 处理后的数据
│   └── raw_data/          # 原始数据
├── config.py               # 配置文件
├── main.py                 # 主应用
├── memory.py               # 记忆系统
├── semantic_analyzer.py    # 语义分析器
├── skill_system.py         # 技能系统
├── rag_system.py           # RAG 系统
├── prompt_assembler.py     # 提示组装器
├── emotion_vector.py       # 情感向量系统
├── llm_client.py           # LLM 客户端
├── reflection.py           # 反思引擎
├── state_machine.py        # 状态机
├── run.py                  # 启动脚本
└── requirements.txt        # 依赖文件
```

## 🚀 快速开始

### 1. 环境准备

#### 安装系统依赖

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y redis-server python3-pip python3-venv

# 启动 Redis
sudo redis-server --daemonize yes
```

#### 安装 Python 依赖

```bash
# 创建虚拟环境 (可选)
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

编辑 `config.py` 文件，根据需要修改配置：

- `SERVER_HOST`: 服务器绑定地址 (默认: 0.0.0.0)
- `SERVER_PORT`: 服务器端口 (默认: 8000)
- `OLLAMA_BASE_URL`: Ollama API 地址 (默认: http://localhost:11434)
- `OLLAMA_MODEL`: 使用的 LLM 模型 (默认: qwen3-coder:30b)
- `DEEPSEEK_API_KEY`: DeepSeek API 密钥
- `VECTOR_DB_PATH`: 向量数据库路径 (默认: ./data/vector_db)

### 3. 启动服务器

#### 方法 1: 使用启动脚本

```bash
# 基本启动
python run.py

# 自定义配置
python run.py --host 0.0.0.0 --port 8000 --reload

# 跳过 Redis 检查
python run.py --no-redis-check
```

#### 方法 2: 直接启动

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 测试 API

```bash
# 测试聊天接口
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "message": "你好", "stream": false}'

# 测试技能接口
curl -X POST http://localhost:8000/api/v1/skill \
  -H "Content-Type: application/json" \
  -d '{"skill_name": "wsl2", "params": {"action": "info"}}'
```

## 🌐 远程访问

### 使用 Cloudflare Tunnel

1. **安装 cloudflared**

```bash
# 下载 cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64

# 赋予执行权限
chmod +x cloudflared-linux-amd64

sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared
```

2. **创建临时通道**

```bash
# 启动临时通道
cloudflared tunnel --url http://localhost:8000

# 输出示例
# 2024-01-01T00:00:00Z INF Registered tunnel connection connIndex=0 connection=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx event=0 ip=198.41.192.57 location=hkg01 protocol=quic
# 2024-01-01T00:00:00Z INF Tunnel ready tunnelURL=https://xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.trycloudflare.com
```

3. **更新客户端配置**

将客户端的 `SERVER_URL` 配置为输出的 tunnelURL。

## 🧠 系统架构

### 核心组件

1. **语义分析器 (SemanticAnalyzer)**
   - 分析用户输入的意图、情感、关键词等
   - 识别 WSL2 命令意图

2. **记忆系统 (MemorySystem)**
   - 工作记忆: 最近的对话历史
   - 长期记忆: 重要的信息和经历
   - 深度记忆: 用户画像和核心记忆

3. **状态机 (StateDecisionEngine)**
   - 根据情境和人格参数决定回应模式
   - 支持专业、哲思、摆烂、恶搞、简洁等模式

4. **提示组装器 (PromptAssembler)**
   - 整合人格、记忆、环境等信息
   - 构建完整的提示给 LLM

5. **情感系统 (EmotionSystem)**
   - 生成情感向量
   - 映射到 Live2D 参数

6. **RAG 系统 (RAGSystem)**
   - 检索专业知识
   - 整合到提示中

7. **技能系统 (SkillSystem)**
   - 执行 WSL2 命令
   - 提供系统信息

### 工作流程

1. **用户输入** → **语义分析** → **状态决策** → **记忆检索** → **提示组装** → **LLM 调用** → **响应生成** → **情感处理** → **记忆更新** → **返回响应**

2. **WSL2 命令** → **意图识别** → **命令检测** → **确认执行** → **技能调用** → **返回结果**

## 📡 API 端点

### 聊天接口
- `POST /api/v1/chat`: 发送聊天消息
- `GET /api/v1/chat`: 获取聊天历史

### 技能接口
- `POST /api/v1/skill`: 调用技能

### 管理接口
- `POST /api/v1/admin`: 执行管理命令

### WebSocket 接口
- `GET /api/v1/ws`: WebSocket 连接

## ⚠️ 注意事项

1. **Redis 依赖**
   - 如果服务器上没有安装 Redis，系统会降级使用内存存储
   - 内存存储在服务器重启后会丢失数据

2. **Ollama 依赖**
   - 系统需要 Ollama 服务运行在 `http://localhost:11434`
   - 确保已安装并启动 Ollama 服务

3. **WSL2 命令执行**
   - 仅支持安全的 WSL2 相关命令
   - 命令执行超时时间默认为 300 秒

4. **性能优化**
   - 对于生产环境，建议使用 `--reload` 参数启动服务器
   - 确保服务器有足够的内存和 CPU 资源

## 🛡️ 安全提示

1. **命令执行安全**
   - 系统会检查 WSL2 命令的安全性
   - 避免执行危险的命令

2. **网络安全**
   - 建议使用 Cloudflare Tunnel 进行远程访问
   - 避免直接暴露服务器到公网

3. **数据安全**
   - 敏感信息不应存储在记忆系统中
   - 定期清理不需要的记忆

## 📞 技术支持

如果在使用过程中遇到问题，请检查以下几点：

1. **依赖问题**：确保所有依赖都已正确安装
2. **Ollama 服务**：确保 Ollama 服务正在运行
3. **Redis 服务**：如果使用 Redis，确保 Redis 服务正在运行
4. **网络连接**：确保客户端能够访问服务器地址
5. **日志信息**：查看服务器日志以获取详细的错误信息

## 📄 许可证

MIT License
