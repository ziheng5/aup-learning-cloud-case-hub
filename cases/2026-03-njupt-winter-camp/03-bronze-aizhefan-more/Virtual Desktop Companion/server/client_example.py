import asyncio
import json
import httpx
import websockets


class ServerClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws")
        self.session_id = None
    
    async def create_session(self) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{self.base_url}/api/v1/session")
            data = response.json()
            self.session_id = data["session_id"]
            return self.session_id
    
    async def send_message(self, message: str, stream: bool = False):
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            payload = {
                "session_id": self.session_id,
                "message": message,
                "stream": stream
            }
            response = await client.post(
                f"{self.base_url}/api/v1/chat",
                json=payload
            )
            return response.json()
    
    async def websocket_chat(self, message: str, stream: bool = False):
        async with websockets.connect(f"{self.ws_url}/api/v1/ws") as websocket:
            await websocket.send(json.dumps({
                "message": message,
                "stream": stream
            }))
            
            if stream:
                full_response = ""
                while True:
                    data = json.loads(await websocket.recv())
                    if data["type"] == "start":
                        print("开始响应...")
                    elif data["type"] == "token":
                        full_response += data["token"]
                        print(data["token"], end="", flush=True)
                    elif data["type"] == "complete":
                        print("\n完成！")
                        return data
            else:
                data = json.loads(await websocket.recv())
                return data
    
    async def get_personality(self):
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/api/v1/personality")
            return response.json()
    
    async def get_memory(self, session_id: str = None):
        session_id = session_id or self.session_id
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/api/v1/memory/{session_id}")
            return response.json()
    
    async def clear_memory(self, session_id: str = None):
        session_id = session_id or self.session_id
        async with httpx.AsyncClient() as client:
            response = await client.delete(f"{self.base_url}/api/v1/memory/{session_id}")
            return response.json()
    
    async def trigger_reflection(self):
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{self.base_url}/api/v1/reflection")
            return response.json()
    
    async def health_check(self):
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/api/v1/health")
            return response.json()


async def main():
    client = ServerClient("http://localhost:8000")
    
    print("? 健康检查...")
    health = await client.health_check()
    print(f"状态: {health}")
    
    print("\n? 创建会话...")
    session_id = await client.create_session()
    print(f"会话ID: {session_id}")
    
    print("\n? 发送消息...")
    response = await client.send_message("你好，介绍一下你自己")
    print(f"回复: {response['text']}")
    print(f"情绪: {response['emotion']}")
    print(f"动作: {response['action']}")
    print(f"Live2D参数: {response['live2d_params']}")
    print(f"元数据: {response['metadata']}")
    
    print("\n? 获取人格参数...")
    personality = await client.get_personality()
    print(f"人格特质: {personality['traits']}")
    
    print("\n? 获取记忆...")
    memory = await client.get_memory()
    print(f"记忆: {memory}")
    
    print("\n? 测试键盘控制...")
    response = await client.send_message("鼠标已经为你准备好了")
    print(f"回复: {response['text']}")
    if response.get('keyboard_command'):
        print(f"键盘指令: {response['keyboard_command']}")
    
    print("\n? 测试完成！")


if __name__ == "__main__":
    asyncio.run(main())
