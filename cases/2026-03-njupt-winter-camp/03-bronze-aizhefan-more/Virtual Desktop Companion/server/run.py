#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Desktop Pet Server 启动脚本
适用于 Linux / Jupyter 服务器环境
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

SERVER_DIR = Path(__file__).parent
DATA_DIR = SERVER_DIR / "data"

def create_directories():
    """创建必要的数据目录"""
    dirs = [
        DATA_DIR,
        DATA_DIR / "vector_db",
        DATA_DIR / "reflection_logs",
        DATA_DIR / "daily_summaries",
        DATA_DIR / "evolution_logs"
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"? 目录: {d}")

def check_redis():
    """检查 Redis 是否可用"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("? Redis 连接成功")
        return True
    except Exception as e:
        print(f"? Redis 连接失败: {e}")
        print("  请确保 Redis 已安装并运行:")
        print("  sudo apt-get install redis-server")
        print("  redis-server --daemonize yes")
        return False

def check_dependencies():
    """检查依赖是否安装"""
    required = [
        'fastapi', 'uvicorn', 'httpx', 'pydantic', 
        'redis', 'chromadb'
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"? 缺少依赖: {missing}")
        print("  正在安装...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 
            str(SERVER_DIR / "requirements.txt")
        ])
    else:
        print("? 所有依赖已安装")

def start_server(host="0.0.0.0", port=8000, reload=False):
    """启动服务器"""
    import uvicorn
    
    print(f"\n? 启动 AI Desktop Pet Server")
    print(f"? 地址: http://{host}:{port}")
    print(f"? API: http://{host}:{port}/api/v1")
    print(f"? 健康检查: http://{host}:{port}/api/v1/health")
    print(f"? 重载模式: {'开启' if reload else '关闭'}\n")
    
    sys.path.insert(0, str(SERVER_DIR))
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

def main():
    parser = argparse.ArgumentParser(description='AI Desktop Pet Server')
    parser.add_argument('--host', default='0.0.0.0', help='绑定地址')
    parser.add_argument('--port', type=int, default=8000, help='端口')
    parser.add_argument('--reload', action='store_true', help='开启自动重载')
    parser.add_argument('--no-redis-check', action='store_true', help='跳过Redis检查')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("  AI Desktop Pet Server - 启动脚本")
    print("=" * 50)
    
    print("\n? 创建数据目录...")
    create_directories()
    
    print("\n? 检查依赖...")
    check_dependencies()
    
    if not args.no_redis_check:
        print("\n? 检查 Redis...")
        check_redis()
    
    print("\n" + "=" * 50)
    
    try:
        start_server(args.host, args.port, args.reload)
    except KeyboardInterrupt:
        print("\n\n? 服务器已停止")
        sys.exit(0)
    except Exception as e:
        print(f"\n? 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
