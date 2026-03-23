# -*- coding: utf-8 -*-
import requests
import json

SERVER_URL = "https://continues-surgeons-sue-lucas.trycloudflare.com"

session = requests.Session()

def test_connection():
    """测试连接"""
    try:
        response = session.get(f"{SERVER_URL}/api/v1/health", timeout=10)
        print(f"✓ 连接成功: {response.json()}")
        return True
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        return False

def send_message(message, session_id=None):
    """发送聊天消息"""
    data = {"message": message}
    if session_id:
        data["session_id"] = session_id
    
    try:
        response = session.post(
            f"{SERVER_URL}/api/v1/chat",
            json=data,
            timeout=30
        )
        result = response.json()
        print(f"\n收到响应:")
        print(f"  session_id: {result.get('session_id')}")
        print(f"  AI回复: {result.get('text')}")
        print(f"  情感: {result.get('emotion')}")
        print(f"  动作: {result.get('action')}")
        if result.get('metadata', {}).get('is_easter_egg'):
            print(f"  [彩蛋触发!]")
        return result
    except Exception as e:
        print(f"? 请求失败: {e}")
        return None

def chat_loop():
    """交互式聊天循环"""
    print("=" * 50)
    print("  AI 桌面宠物 - 客户端")
    print("=" * 50)
    print(f"\n服务器地址: {SERVER_URL}")
    
    if not test_connection():
        return
    
    session_id = None
    print("\n输入消息开始聊天 (输入 'quit' 退出)")
    print("彩蛋测试: 先输入 '你好'，再输入 '什么？'\n")
    
    while True:
        message = input("\n你: ").strip()
        if message.lower() in ['quit', 'exit', '退出', 'q']:
            print("再见!")
            break
        if not message:
            continue
        
        result = send_message(message, session_id)
        if result:
            session_id = result.get('session_id')

if __name__ == "__main__":
    chat_loop()
