# AI Assistant Project Overview

## Project Summary

This project consists of two main components:
1. **LLM (Large Language Model)** - A 3B parameter dialogue language model optimized for AMD ROCm
2. **Virtual Desktop Companion** - An AI desktop pet system with personality simulation and WSL2 command execution capabilities

## Project Structure

```
├── llm/                # 3B parameter dialogue language model
│   ├── configs/        # Configuration files
│   ├── core/           # Core model architecture
│   ├── data/           # Data processing
│   ├── inference/      # Inference system
│   ├── scripts/        # Training and inference scripts
│   └── training/       # Training system
├── Virtual Desktop Companion/  # AI desktop pet system
│   ├── client/         # Client-side code
│   └── server/         # Server-side code
├── README.md           # English documentation
├── README_ZH.md        # Chinese documentation
├── requirements.txt    # Dependencies
├── main.ipynb          # English notebook
└── main_zh.ipynb       # Chinese notebook
```

## LLM Component

### Model Architecture

- **Model Size**: 3B parameters
- **Architecture**: Decoder-only Transformer (GPT-like)
- **Hidden Dimension**: 3072
- **Layers**: 32
- **Attention Heads**: 24 (GQA: 8 KV heads)
- **Intermediate Size**: 8192
- **Maximum Sequence Length**: 4096
- **Vocabulary Size**: 65536

### Key Technologies

1. **FlashAttention 2**: Efficient attention computation, saves memory
2. **RoPE Position Encoding**: Rotational position encoding, supports long sequences
3. **SwiGLU Activation**: Gated linear unit, better performance than GELU
4. **RMSNorm**: Efficient normalization
5. **GQA Grouped Query Attention**: Balances performance and quality between MQA and MHA

### Hardware Requirements

- **GPU**: AMD Radeon 8060S (RDNA 3.5, gfx1151) or higher
- **Memory**: 64GB shared memory (minimum 16GB)
- **CPU**: 4+ cores
- **RAM**: 16GB+
- **ROCm**: 7.10+

## Virtual Desktop Companion Component

### System Architecture

- **Client-Server Architecture**: Separate client and server components
- **Client**: Provides user interface, handles user input, displays AI responses
- **Server**: Processes client requests, executes LLM inference, manages knowledge base

### Core Features

1. **Personality Simulation**: AI desktop pet with distinct personality and emotional states
2. **Knowledge Retrieval**: RAG system for专业 knowledge about WSL2
3. **WSL2 Command Execution**: Executes WSL2 commands through local server
4. **Multi-mode Interaction**: Different interaction modes for different scenarios
5. **Memory System**: Tracks conversation history and builds user profile

### Key Components

- **Client Application**: Vue.js-based frontend with Live2D character animation
- **Local Server**: Node.js server for WSL2 command execution
- **Server API**: FastAPI-based backend for LLM inference
- **RAG System**: Manages professional knowledge and provides retrieval services
- **Prompt Assembler**: Builds LLM prompts and manages response strategies
- **Memory System**: Manages session history and builds user profile

## Technical Highlights

1. **AMD ROCm Optimization**: Leveraging AMD GPU capabilities for efficient model training and inference
2. **FlashAttention 2**: Optimized attention mechanism for better memory usage
3. **RAG Technology**: Enhanced knowledge capabilities through retrieval-augmented generation
4. **Multi-API Integration**: Combining different LLM APIs for optimal performance
5. **Local Command Execution**: Safe execution of WSL2 commands through local server
6. **Personality Simulation**: Rich personality and emotional states for engaging interaction

## Installation and Setup

### LLM Component

```bash
# Install dependencies
pip install -r llm/requirements.txt

# Install ROCm version of PyTorch
pip install torch --index-url https://download.pytorch.org/whl/rocm7.1

# Train the model
python llm/scripts/train.py --train_data ./data/train --eval_data ./data/eval

# Run inference
python llm/scripts/inference.py --model_path ./checkpoints/final --prompt "Hello"
```

### Virtual Desktop Companion

```bash
# Server setup
cd Virtual Desktop Companion/server
pip install -r requirements.txt
python main.py

# Client setup
cd Virtual Desktop Companion/client
npm install
node local-server.js
# Open index.html in browser
```

## Use Cases

1. **Personal AI Assistant**: Engage in natural conversations with the AI desktop pet
2. **Technical Support**: Get help with WSL2 commands and troubleshooting
3. **Knowledge Retrieval**: Access information about WSL2 and related technologies
4. **Command Execution**: Execute WSL2 commands through natural language
5. **Personality Interaction**: Experience different interaction modes based on context

## Future Development

1. **Knowledge Base Expansion**: Cover more technical domains and topics
2. **Personality Enhancement**: Deepen personality simulation and emotional intelligence
3. **Security Optimization**: Improve command execution safety and efficiency
4. **Cross-Platform Support**: Extend to more operating systems and environments
5. **Personalization Options**: Provide more customization features for users

## Conclusion

This project demonstrates the integration of advanced LLM technology with practical applications, creating a comprehensive AI assistant system that combines personality simulation, knowledge retrieval, and command execution capabilities. By leveraging AMD ROCm optimization, the system achieves efficient performance while providing engaging and useful interactions for users.
