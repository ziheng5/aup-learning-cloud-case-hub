from pydantic_settings import BaseSettings
from typing import Optional, List, Dict, Any
import os


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AI Desktop Pet Server"
    
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000
    
    OLLAMA_BASE_URL: str = "http://open-webui-ollama.open-webui:11434"
    OLLAMA_MODEL: str = "qwen3-coder:30b"
    OLLAMA_NUM_CTX: int = 32768
    OLLAMA_TEMPERATURE: float = 0.7
    
    DEEPSEEK_API_KEY: str = "##############################"  # 这个API主要负责清洗数据和处理复杂问题任务
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-chat"
    USE_DEEPSEEK_FOR_COMPLEX: bool = True
    
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    VECTOR_DB_PATH: str = "./data/vector_db"
    MEMORY_DB_PATH: str = "./data/memory.db"
    
    WORKING_MEMORY_SIZE: int = 10
    LONG_TERM_MEMORY_TOP_K: int = 5
    DEEP_MEMORY_THRESHOLD: float = 0.8
    
    ENABLE_REFLECTION: bool = True
    REFLECTION_INTERVAL_SECONDS: int = 300
    
    MEMORY_DECAY_RATE: float = 0.01
    MEMORY_FUZZY_FACTOR: float = 0.2
    
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()


class PersonalityConfig:
    PET_NAME: str = "小助手"
    PET_IDENTITY: str = "一个可爱的AI桌面宠物"
    
    BASE_TRAITS: Dict[str, float] = {
        "conciseness": 8.5,
        "philosophical": 7.2,
        "playfulness": 4.5,
        "warmth": 6.8,
        "self_awareness": 9.0,
        "slacking_tendency": 3.5
    }
    
    TRAIT_MIN: float = 0.0
    TRAIT_MAX: float = 10.0
    TRAIT_ADJUST_RATE: float = 0.05
    
    MOOD_PERSISTENCE: float = 0.7


class StateModes:
    PROFESSIONAL = "professional"
    PHILOSOPHICAL = "philosophical"
    SLACKING = "slacking"
    MEME = "meme"
    CONCISE = "concise"
    
    ALL_MODES = [PROFESSIONAL, PHILOSOPHICAL, SLACKING, MEME, CONCISE]


class IntentTypes:
    HELP = "help"
    CHAT = "chat"
    PHILOSOPHICAL = "philosophical"
    ABSURD = "absurd"
    REPETITION = "repetition"
    UNCLEAR = "unclear"
    KEYBOARD = "keyboard"
    
    ALL_INTENTS = [HELP, CHAT, PHILOSOPHICAL, ABSURD, REPETITION, UNCLEAR, KEYBOARD]


class EmotionLabels:
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    NEUTRAL = "neutral"
    THINKING = "thinking"
    EXCITED = "excited"
    TIRED = "tired"
    ANXIOUS = "anxious"
    
    ALL_EMOTIONS = [HAPPY, SAD, ANGRY, NEUTRAL, THINKING, EXCITED, TIRED, ANXIOUS]


class ActionTypes:
    WAVE = "wave"
    NOD = "nod"
    THINK = "think"
    SLEEP = "sleep"
    HAPPY = "happy"
    SAD = "sad"
    IDLE = "idle"
    LOVE = "love"
    
    ALL_ACTIONS = [WAVE, NOD, THINK, SLEEP, HAPPY, SAD, IDLE, LOVE]


KEYBOARD_TRIGGER_PHRASES = [
    "鼠标已经为你准备好了",
    "键盘给你",
    "你来操作",
    "帮我打字",
    "帮我输入",
    "控制键盘",
    "键盘已就绪",
    "可以帮我输入",
]


class SeriousTopicTypes:
    MEDICAL = "medical"
    LEGAL = "legal"
    FINANCIAL = "financial"
    CRISIS = "crisis"
    SAFETY = "safety"
    EMOTIONAL_CRISIS = "emotional_crisis"
    
    ALL_TYPES = [MEDICAL, LEGAL, FINANCIAL, CRISIS, SAFETY, EMOTIONAL_CRISIS]


class AtmosphereTypes:
    LIGHT = "light"
    SERIOUS = "serious"
    MELANCHOLY = "melancholy"
    TENSE = "tense"
    NEUTRAL = "neutral"
    
    ALL_TYPES = [LIGHT, SERIOUS, MELANCHOLY, TENSE, NEUTRAL]
