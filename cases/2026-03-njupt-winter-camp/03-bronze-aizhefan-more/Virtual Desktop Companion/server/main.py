# -*- coding: utf-8 -*-
import asyncio
import json
import uuid
import time
import os
import sys
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

os.environ['PYTHONIOENCODING'] = 'utf-8'

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from config import settings, StateModes, IntentTypes, ActionTypes, PersonalityConfig
from memory import MemorySystem
from semantic_analyzer import SemanticAnalyzer
from state_machine import StateDecisionEngine, PersonalityCore
from prompt_assembler import PromptAssembler
from llm_client import LLMClient
from reflection import ReflectionEngine
from emotion_vector import EmotionSystem
from rag_system import rag_system
from skill_system import skill_system


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    stream: bool = False
    mode: Optional[str] = "chat"  # chat 或 wsl2


class ChatResponse(BaseModel):
    session_id: str
    text: str
    emotion: str
    action: str
    emotion_vector: Dict[str, float]
    live2d_params: Dict[str, Any]
    keyboard_command: Optional[Dict[str, Any]] = None
    wsl2_command: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any]


class AdminCommandRequest(BaseModel):
    command: str
    params: Optional[Dict[str, Any]] = None


class PersonalityUpdateRequest(BaseModel):
    trait: str
    value: float


class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
    
    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "created_at": time.time(),
            "last_active": time.time(),
            "history": []
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        if session_id not in self.sessions:
            return None
        self.sessions[session_id]["last_active"] = time.time()
        return self.sessions[session_id]
    
    def add_to_history(self, session_id: str, message: Dict[str, Any]):
        if session_id in self.sessions:
            self.sessions[session_id]["history"].append(message)
            if len(self.sessions[session_id]["history"]) > 100:
                self.sessions[session_id]["history"] = self.sessions[session_id]["history"][-50:]


memory_system = MemorySystem()
semantic_analyzer = SemanticAnalyzer()
state_engine = StateDecisionEngine()
personality_core = PersonalityCore()
prompt_assembler = PromptAssembler()
llm_client = LLMClient()
reflection_engine = ReflectionEngine(memory_system, personality_core, state_engine)
emotion_system = EmotionSystem()
session_manager = SessionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("? AI Desktop Pet Server 启动中...")
    yield
    print("? 服务器已关闭")


app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/api/v1/chat",
            "ws": "/api/v1/ws",
            "session": "/api/v1/session",
            "admin": "/api/v1/admin",
        }
    }


@app.post(f"{settings.API_V1_STR}/session")
async def create_session():
    session_id = session_manager.create_session()
    return {"session_id": session_id}


def handle_admin_command(command_type: str, params: Dict[str, Any], message: str) -> Dict[str, Any]:
    """处理管理员命令"""
    
    if command_type == "get_trait":
        traits = personality_core.get_traits()
        return {
            "success": True,
            "type": "get_trait",
            "data": traits,
            "message": f"当前人格参数：\n" + "\n".join([f"- {k}: {v:.1f}/10" for k, v in traits.items()])
        }
    
    elif command_type == "set_trait":
        if params.get("reset_all"):
            personality_core.reset()
            return {
                "success": True,
                "type": "reset",
                "message": "人格参数已重置为默认值"
            }
        
        trait = params.get("trait")
        value = params.get("value")
        
        if not trait or value is None:
            return {
                "success": False,
                "type": "set_trait",
                "message": "请指定要设置的参数，格式：设置人格参数：简洁=8"
            }
        
        if trait not in PersonalityConfig.BASE_TRAITS:
            return {
                "success": False,
                "type": "set_trait",
                "message": f"未知参数：{trait}，可选参数：{', '.join(PersonalityConfig.BASE_TRAITS.keys())}"
            }
        
        personality_core.set_trait(trait, value)
        return {
            "success": True,
            "type": "set_trait",
            "trait": trait,
            "value": value,
            "message": f"已将 {trait} 设置为 {value}/10"
        }
    
    elif command_type == "get_mode":
        return {
            "success": True,
            "type": "get_mode",
            "message": f"当前可用模式：{', '.join(StateModes.ALL_MODES)}"
        }
    
    elif command_type == "set_mode":
        mode = params.get("mode")
        if mode and mode in StateModes.ALL_MODES:
            personality_core.force_mode = mode
            return {
                "success": True,
                "type": "set_mode",
                "mode": mode,
                "message": f"已强制使用 {mode} 模式"
            }
        return {
            "success": False,
            "message": "无效的模式名称"
        }
    
    elif command_type == "set_memory":
        content = params.get("content") or params.get("value")
        if content:
            memory_system.long_term.add(
                content=content,
                importance=0.9,
                emotion_tag="neutral",
                tags=["admin", "manual"],
                source="admin"
            )
            return {
                "success": True,
                "type": "set_memory",
                "message": f"已保存记忆：{content[:50]}..."
            }
        return {
            "success": False,
            "message": "请指定要保存的内容"
        }
    
    elif command_type == "get_memory":
        recent = memory_system.long_term.get_all()[:10]
        return {
            "success": True,
            "type": "get_memory",
            "data": recent,
            "message": f"共有 {len(recent)} 条记忆"
        }
    
    elif command_type == "reset":
        personality_core.reset()
        return {
            "success": True,
            "type": "reset",
            "message": "已重置所有人格参数"
        }
    
    elif command_type == "help":
        help_text = """
? 可用管理命令：

? 人格参数管理：
- 查看人格参数 / 显示人格 / get trait
- 设置人格参数：简洁=8 / 哲思=7 / trait=value
- 重置人格 / reset

? 模式控制：
- 当前模式 / get mode
- 进入专业模式 / 强制模式=专业

? 记忆管理：
- 记住xxx / 保存xxx
- 查看记忆 / get memory

? 帮助：
- 帮助 / help / ?
        """
        return {
            "success": True,
            "type": "help",
            "message": help_text
        }
    
    return {
        "success": False,
        "message": f"未知命令类型：{command_type}"
    }


easter_egg_state = {}

def check_easter_egg(user_message: str, session_id: str) -> Optional[Dict[str, Any]]:
    normalized = user_message.strip().lower()
    
    if normalized in ["你好", "您好", "hi", "hello", "嗨"]:
        easter_egg_state[session_id] = {"triggered": True, "step": 1}
        return {
            "text": "不好",
            "emotion": "playful",
            "action": "mischief"
        }
    
    if session_id in easter_egg_state:
        state = easter_egg_state[session_id]
        if state.get("step") == 1:
            confusion_patterns = ["什么", "为什么", "啥", "怎么", "？", "?", "疑惑", "不解", "什么意思", "啥意思"]
            if any(p in normalized for p in confusion_patterns):
                easter_egg_state[session_id] = {"triggered": True, "step": 2}
                return {
                    "text": "开玩笑的，你好~",
                    "emotion": "happy",
                    "action": "laugh"
                }
    
    return None


@app.post(f"{settings.API_V1_STR}/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id or session_manager.create_session()
    
    session = session_manager.get_session(session_id)
    if not session:
        session_id = session_manager.create_session()
    
    easter_egg_result = check_easter_egg(request.message, session_id)
    if easter_egg_result:
        emotion_data = {"label": easter_egg_result["emotion"], "valence": 0.6, "arousal": 0.6}
        emotion_result = emotion_system.process(emotion_data, easter_egg_result["action"])
        
        return ChatResponse(
            session_id=session_id,
            text=easter_egg_result["text"],
            emotion=easter_egg_result["emotion"],
            action=easter_egg_result["action"],
            emotion_vector=emotion_result["emotion_vector"],
            live2d_params=emotion_result["live2d_params"],
            keyboard_command=None,
            metadata={"is_easter_egg": True}
        )
    
    situation = semantic_analyzer.analyze(session_id, request.message)
    
    if situation.is_admin_command:
        params = semantic_analyzer.get_admin_params(request.message)
        admin_result = handle_admin_command(situation.admin_command_type, params, request.message)
        
        emotion_data = {"label": "neutral", "valence": 0.5, "arousal": 0.5}
        emotion_result = emotion_system.process(emotion_data, "idle")
        
        return ChatResponse(
            session_id=session_id,
            text=admin_result.get("message", "命令已执行"),
            emotion="neutral",
            action="idle",
            emotion_vector=emotion_result["emotion_vector"],
            live2d_params=emotion_result["live2d_params"],
            keyboard_command=None,
            metadata={
                "is_admin": True,
                "admin_type": situation.admin_command_type,
                "admin_data": admin_result.get("data")
            }
        )
    
    # 处理WSL2命令意图
    if situation.intent == "wsl2_command" or request.mode == "wsl2":
        # WSL2模式下，使用LLM来理解用户意图并生成命令
        environment = semantic_analyzer.get_environment_context()
        
        personality_traits = personality_core.get_traits()
        decision = state_engine.decide(situation, personality_traits, environment)
        
        memory_context = memory_system.retrieve_context(session_id, request.message)
        
        # 构建专门用于WSL2命令生成的prompt
        wsl2_prompt = f"""
你是一个WSL2命令专家，需要根据用户的自然语言请求生成具体的WSL2命令。

用户请求：{request.message}

请分析用户的请求，生成一个具体的WSL2命令来完成用户的需求。

输出格式：
{
  "command": "具体的WSL2命令",
  "args": ["参数1", "参数2"],
  "description": "命令的详细描述"
}

例如：
用户请求：查看WSL版本
输出：
{
  "command": "wsl --version",
  "args": ["--version"],
  "description": "查看WSL的版本信息"
}

请严格按照上述格式输出，不要有任何额外内容！
"""
        
        try:
            # 使用DeepSeek API来生成命令
            llm_response = await llm_client.generate(
                prompt=wsl2_prompt,
                complexity=0.7,
                stream=False,
                mode="wsl2"
            )
            
            # 解析LLM返回的命令
            parsed_data = llm_response.parsed_data or {}
            wsl2_command = parsed_data.get("wsl2_command")
            
            if not wsl2_command:
                # 尝试直接解析JSON
                try:
                    json_start = llm_response.text.find("{")
                    json_end = llm_response.text.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = llm_response.text[json_start:json_end]
                        wsl2_command = json.loads(json_str)
                except (json.JSONDecodeError, ValueError):
                    pass
            
            if wsl2_command and "command" in wsl2_command:
                # 如果成功生成命令，返回结果
                emotion_data = {"label": "neutral", "valence": 0.6, "arousal": 0.5}
                emotion_result = emotion_system.process(emotion_data, "idle")
                
                return ChatResponse(
                    session_id=session_id,
                    text=f"我理解您想要：{wsl2_command.get('description', '执行WSL2命令')}",
                    emotion="neutral",
                    action="idle",
                    emotion_vector=emotion_result["emotion_vector"],
                    live2d_params=emotion_result["live2d_params"],
                    keyboard_command=None,
                    metadata={
                        "is_wsl2_intent": True,
                        "intent_confidence": situation.intent_confidence
                    },
                    wsl2_command=wsl2_command
                )
            else:
                # 如果没有生成有效的命令，返回错误信息
                emotion_data = {"label": "neutral", "valence": 0.5, "arousal": 0.5}
                emotion_result = emotion_system.process(emotion_data, "idle")
                
                return ChatResponse(
                    session_id=session_id,
                    text="抱歉，我无法理解您的WSL2相关请求，请尝试更明确地描述您的需求。",
                    emotion="neutral",
                    action="idle",
                    emotion_vector=emotion_result["emotion_vector"],
                    live2d_params=emotion_result["live2d_params"],
                    keyboard_command=None,
                    metadata={
                        "is_wsl2_intent": True,
                        "intent_confidence": situation.intent_confidence
                    }
                )
        except Exception as e:
            # 如果LLM调用失败，返回错误信息
            emotion_data = {"label": "neutral", "valence": 0.5, "arousal": 0.5}
            emotion_result = emotion_system.process(emotion_data, "idle")
            
            return ChatResponse(
                session_id=session_id,
                text=f"生成WSL2命令失败：{str(e)}",
                emotion="neutral",
                action="idle",
                emotion_vector=emotion_result["emotion_vector"],
                live2d_params=emotion_result["live2d_params"],
                keyboard_command=None,
                metadata={
                    "is_wsl2_intent": True,
                    "intent_confidence": situation.intent_confidence,
                    "error": str(e)
                }
            )
    
    environment = semantic_analyzer.get_environment_context()
    
    personality_traits = personality_core.get_traits()
    decision = state_engine.decide(situation, personality_traits, environment)
    
    memory_context = memory_system.retrieve_context(session_id, request.message)
    
    assembled_prompt = prompt_assembler.assemble(
        user_input=request.message,
        personality=personality_core,
        mode=decision.mode,
        memory_context=memory_context,
        environment=environment,
        keywords=situation.keywords,
        situation={
            "is_serious_topic": situation.is_serious_topic,
            "atmosphere": situation.atmosphere,
            "is_repetition": situation.is_repetition
        },
        is_keyboard_request=situation.is_keyboard_request
    )
    
    try:
        llm_response = await llm_client.generate(
            prompt=assembled_prompt.full_prompt,
            complexity=situation.complexity,
            stream=request.stream,
            mode=request.mode
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM调用失败: {str(e)}")
    
    parsed_data = llm_response.parsed_data or {}
    text = parsed_data.get("text", llm_response.text)
    emotion = parsed_data.get("emotion", "neutral")
    action = parsed_data.get("action", "idle")
    
    if situation.is_keyboard_request:
        keyboard_command = parsed_data.get("keyboard_command", {
            "type": "type_text",
            "text": "好的，我来帮你输入"
        })
        text = "好的，我准备好了！"
        emotion = "ready"
        action = "ready"
    else:
        keyboard_command = None
    
    emotion_data = {
        "label": emotion,
        "valence": 0.5,
        "arousal": 0.5
    }
    emotion_result = emotion_system.process(emotion_data, action)
    
    memory_operation = parsed_data.get("memory_operation", {"should_store": False, "importance": 0.5, "tags": []})
    if memory_operation.get("should_store"):
        importance = memory_operation.get("importance", 0.5)
        tags = memory_operation.get("tags", [])
        memory_system.add_interaction(
            session_id, request.message, text, importance, emotion, tags
        )
    
    personality_impact = parsed_data.get("personality_impact", 0.0)
    if abs(personality_impact) > 0.01:
        for trait in personality_traits:
            personality_core.update_trait(trait, personality_impact * 0.1)
    
    reflection_engine.log_interaction({
        "session_id": session_id,
        "user_input": request.message,
        "ai_response": text,
        "emotion": emotion,
        "mode": decision.mode,
        "keywords": situation.keywords,
        "importance": memory_operation.get("importance", 0.5)
    })
    
    response_time = llm_response.duration_ms / 1000 if llm_response.duration_ms else 1.0
    reflection_engine.process_immediate_feedback(
        user_message=request.message,
        ai_response=text,
        mode=decision.mode,
        intent=situation.intent,
        emotion=situation.emotion.get("label", "neutral"),
        response_time=response_time
    )
    
    if settings.ENABLE_REFLECTION:
        asyncio.create_task(reflection_engine.run_reflection())
    
    return ChatResponse(
        session_id=session_id,
        text=text,
        emotion=emotion,
        action=action,
        emotion_vector=emotion_result["emotion_vector"],
        live2d_params=emotion_result["live2d_params"],
        keyboard_command=keyboard_command,
        metadata={
            "mode": decision.mode,
            "model_used": llm_response.model,
            "duration_ms": llm_response.duration_ms,
            "tokens_used": llm_response.tokens_used,
            "decision_reasoning": decision.reasoning
        }
    )


@app.websocket(f"{settings.API_V1_STR}/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    session_id = session_manager.create_session()
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue
            
            message = message_data.get("message", "")
            stream = message_data.get("stream", False)
            
            easter_egg_result = check_easter_egg(message, session_id)
            if easter_egg_result:
                emotion_data = {"label": easter_egg_result["emotion"], "valence": 0.6, "arousal": 0.6}
                emotion_result = emotion_system.process(emotion_data, easter_egg_result["action"])
                
                await websocket.send_json({
                    "type": "response",
                    "session_id": session_id,
                    "text": easter_egg_result["text"],
                    "emotion": easter_egg_result["emotion"],
                    "action": easter_egg_result["action"],
                    "emotion_vector": emotion_result["emotion_vector"],
                    "live2d_params": emotion_result["live2d_params"],
                    "keyboard_command": None,
                    "metadata": {"is_easter_egg": True}
                })
                continue
            
            situation = semantic_analyzer.analyze(session_id, message)
            
            if situation.is_admin_command:
                params = semantic_analyzer.get_admin_params(message)
                admin_result = handle_admin_command(situation.admin_command_type, params, message)
                
                emotion_data = {"label": "neutral", "valence": 0.5, "arousal": 0.5}
                emotion_result = emotion_system.process(emotion_data, "idle")
                
                await websocket.send_json({
                    "type": "admin_response",
                    "session_id": session_id,
                    "text": admin_result.get("message", "命令已执行"),
                    "emotion": "neutral",
                    "action": "idle",
                    "emotion_vector": emotion_result["emotion_vector"],
                    "live2d_params": emotion_result["live2d_params"],
                    "metadata": {
                        "is_admin": True,
                        "admin_type": situation.admin_command_type,
                        "admin_data": admin_result.get("data")
                    }
                })
                continue
            
            environment = semantic_analyzer.get_environment_context()
            
            personality_traits = personality_core.get_traits()
            decision = state_engine.decide(situation, personality_traits, environment)
            
            memory_context = memory_system.retrieve_context(session_id, message)
            
            assembled_prompt = prompt_assembler.assemble(
                user_input=message,
                personality=personality_core,
                mode=decision.mode,
                memory_context=memory_context,
                environment=environment,
                keywords=situation.keywords,
                situation={
                    "is_serious_topic": situation.is_serious_topic,
                    "atmosphere": situation.atmosphere,
                    "is_repetition": situation.is_repetition
                },
                is_keyboard_request=situation.is_keyboard_request
            )
            
            if stream:
                await websocket.send_json({
                    "type": "start",
                    "session_id": session_id
                })
                
                full_response = ""
                async for token in llm_client.generate_stream(
                    prompt=assembled_prompt.full_prompt,
                    complexity=situation.complexity
                ):
                    full_response += token
                    await websocket.send_json({
                        "type": "token",
                        "token": token
                    })
                
                try:
                    json_start = full_response.find("{")
                    json_end = full_response.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = full_response[json_start:json_end]
                        parsed_data = json.loads(json_str)
                    else:
                        parsed_data = {}
                except:
                    parsed_data = {}
                
                text = parsed_data.get("text", full_response)
                emotion = parsed_data.get("emotion", "neutral")
                action = parsed_data.get("action", "idle")
                
                if situation.is_keyboard_request:
                    keyboard_command = parsed_data.get("keyboard_command", {
                        "type": "type_text",
                        "text": "好的，我来帮你输入"
                    })
                    text = "好的，我准备好了！"
                    emotion = "ready"
                    action = "ready"
                else:
                    keyboard_command = None
                
                emotion_data = {"label": emotion, "valence": 0.5, "arousal": 0.5}
                emotion_result = emotion_system.process(emotion_data, action)
                
                await websocket.send_json({
                    "type": "complete",
                    "session_id": session_id,
                    "text": text,
                    "emotion": emotion,
                    "action": action,
                    "emotion_vector": emotion_result["emotion_vector"],
                    "live2d_params": emotion_result["live2d_params"],
                    "keyboard_command": keyboard_command,
                    "metadata": {
                        "mode": decision.mode,
                        "decision_reasoning": decision.reasoning
                    }
                })
            else:
                try:
                    llm_response = await llm_client.generate(
                        prompt=assembled_prompt.full_prompt,
                        complexity=situation.complexity,
                        mode=message_data.get("mode", "chat")
                    )
                    
                    parsed_data = llm_response.parsed_data or {}
                    text = parsed_data.get("text", llm_response.text)
                    emotion = parsed_data.get("emotion", "neutral")
                    action = parsed_data.get("action", "idle")
                    
                    if situation.is_keyboard_request:
                        keyboard_command = parsed_data.get("keyboard_command")
                        text = "好的，我准备好了！"
                        emotion = "ready"
                        action = "ready"
                    else:
                        keyboard_command = None
                    
                    emotion_data = {"label": emotion, "valence": 0.5, "arousal": 0.5}
                    emotion_result = emotion_system.process(emotion_data, action)
                    
                    memory_operation = parsed_data.get("memory_operation", {"should_store": False})
                    if memory_operation.get("should_store"):
                        importance = memory_operation.get("importance", 0.5)
                        tags = memory_operation.get("tags", [])
                        memory_system.add_interaction(
                            session_id, message, text, importance, emotion, tags
                        )
                    
                    await websocket.send_json({
                        "type": "response",
                        "session_id": session_id,
                        "text": text,
                        "emotion": emotion,
                        "action": action,
                        "emotion_vector": emotion_result["emotion_vector"],
                        "live2d_params": emotion_result["live2d_params"],
                        "keyboard_command": keyboard_command,
                        "metadata": {
                            "mode": decision.mode,
                            "model_used": llm_response.model,
                            "duration_ms": llm_response.duration_ms,
                            "tokens_used": llm_response.tokens_used,
                            "decision_reasoning": decision.reasoning
                        }
                    })
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
    
    except WebSocketDisconnect:
        print(f"Session {session_id} disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


@app.get(f"{settings.API_V1_STR}/personality")
async def get_personality():
    return {
        "traits": personality_core.get_traits(),
        "mood": personality_core.get_mood(),
        "habits": personality_core.habits,
        "force_mode": getattr(personality_core, 'force_mode', None)
    }


@app.post(f"{settings.API_V1_STR}/personality")
async def update_personality(request: PersonalityUpdateRequest):
    if request.trait not in PersonalityConfig.BASE_TRAITS:
        raise HTTPException(status_code=400, detail=f"Unknown trait: {request.trait}")
    
    if request.value < 0 or request.value > 10:
        raise HTTPException(status_code=400, detail="Value must be between 0 and 10")
    
    personality_core.set_trait(request.trait, request.value)
    
    return {
        "success": True,
        "trait": request.trait,
        "value": request.value,
        "traits": personality_core.get_traits()
    }


@app.post(f"{settings.API_V1_STR}/personality/reset")
async def reset_personality():
    personality_core.reset()
    return {
        "success": True,
        "message": "Personality reset to defaults",
        "traits": personality_core.get_traits()
    }


@app.get(f"{settings.API_V1_STR}/memory/{{session_id}}")
async def get_memory(session_id: str):
    context = memory_system.retrieve_context(session_id, "")
    return context


@app.delete(f"{settings.API_V1_STR}/memory/{{session_id}}")
async def clear_memory(session_id: str):
    memory_system.clear_session(session_id)
    return {"status": "success", "message": "Memory cleared"}


@app.get(f"{settings.API_V1_STR}/feedback/stats")
async def get_feedback_stats(window_minutes: int = 30):
    stats = reflection_engine.get_feedback_statistics(window_minutes)
    return stats


@app.post(f"{settings.API_V1_STR}/feedback/apply")
async def apply_feedback_adjustments():
    result = reflection_engine.apply_feedback_based_adjustments()
    return result


@app.get(f"{settings.API_V1_STR}/feedback/weights")
async def get_weight_stats():
    if reflection_engine.weight_adjuster:
        return reflection_engine.weight_adjuster.get_weight_statistics()
    return {"error": "Weight adjuster not available"}


@app.post(f"{settings.API_V1_STR}/admin/command")
async def admin_command(request: AdminCommandRequest):
    """直接执行管理命令"""
    result = handle_admin_command(request.command, request.params or {}, "")
    return result


@app.post(f"{settings.API_V1_STR}/reflection")
async def trigger_reflection():
    result = await reflection_engine.run_reflection(force=True)
    if result:
        return {
            "status": "success",
            "insights": result.insights,
            "memory_updates": len(result.memory_updates),
            "personality_adjustments": len(result.personality_adjustments)
        }
    return {"status": "success", "message": "No reflection needed"}


class SkillRequest(BaseModel):
    skill_name: str
    params: Dict[str, Any] = {}


@app.post(f"{settings.API_V1_STR}/skill")
async def execute_skill(request: SkillRequest):
    """执行技能（MCP接口）"""
    result = await skill_system.execute_skill(request.skill_name, request.params)
    return {
        "success": result.success,
        "data": result.data,
        "message": result.message,
        "skill_name": result.skill_name,
        "execution_time": result.execution_time
    }


@app.get(f"{settings.API_V1_STR}/skill/list")
async def list_skills():
    """列出所有可用技能"""
    skills = skill_system.list_skills()
    return {"skills": skills}


@app.get(f"{settings.API_V1_STR}/skill/{{skill_name}}")
async def get_skill_info(skill_name: str):
    """获取技能信息"""
    info = skill_system.get_skill_info(skill_name)
    if not info:
        raise HTTPException(status_code=404, detail=f"Skill {skill_name} not found")
    return info


@app.get(f"{settings.API_V1_STR}/config")
async def get_config():
    return {
        "pet_name": PersonalityConfig.PET_NAME,
        "pet_identity": PersonalityConfig.PET_IDENTITY,
        "base_traits": PersonalityConfig.BASE_TRAITS,
        "state_modes": StateModes.ALL_MODES,
        "emotion_labels": EmotionLabels.ALL_EMOTIONS,
        "action_types": ActionTypes.ALL_ACTIONS
    }


@app.get(f"{settings.API_V1_STR}/health")
async def health_check():
    return {
        "status": "healthy",
        "ollama_url": settings.OLLAMA_BASE_URL,
        "model": settings.OLLAMA_MODEL,
        "deepseek_enabled": settings.USE_DEEPSEEK_FOR_COMPLEX
    }


if __name__ == "__main__":
    uvicorn.run(
        "server.main:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=True
    )
