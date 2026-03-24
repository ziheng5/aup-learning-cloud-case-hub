import re
import json
import time
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

from config import IntentTypes, EmotionLabels, KEYBOARD_TRIGGER_PHRASES


@dataclass
class SituationPackage:
    intent: str
    intent_confidence: float
    emotion: Dict[str, Any]
    keywords: List[str]
    is_repetition: bool
    repetition_confidence: float
    complexity: float
    is_keyboard_request: bool
    is_serious_topic: bool
    serious_topic_type: Optional[str]
    atmosphere: str
    raw_text: str
    is_admin_command: bool = False
    admin_command_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
            "emotion": self.emotion,
            "keywords": self.keywords,
            "is_repetition": self.is_repetition,
            "repetition_confidence": self.repetition_confidence,
            "complexity": self.complexity,
            "is_keyboard_request": self.is_keyboard_request,
            "is_serious_topic": self.is_serious_topic,
            "serious_topic_type": self.serious_topic_type,
            "atmosphere": self.atmosphere,
            "raw_text": self.raw_text,
            "is_admin_command": self.is_admin_command,
            "admin_command_type": self.admin_command_type
        }


class AdminCommandDetector:
    def __init__(self):
        self.admin_prefixes = ["/admin", "/管理", "/设置", "/调整", "/config", "/set"]
        
        self.command_patterns = {
            "set_trait": [
                r"设置?人格(参数|特质)?[：:](.+)",
                r"调整?人格(参数|特质)?[：:](.+)",
                r"(修改|改变)(人格|性格)(参数|特质)?[：:](.+)",
                r"set.*trait[=:](.+)",
                r"trait[=:](.+)",
            ],
            "get_trait": [
                r"查看?人格(参数|特质)?",
                r"显示?人格(参数|特质)?",
                r"当前人格(参数|特质)?",
                r"get.*trait",
                r"show.*trait",
            ],
            "set_mode": [
                r"设置?模式[=:](.+)",
                r"强制?模式[=:](.+)",
                r"mode[=:](.+)",
                r"进入(.+)模式",
            ],
            "get_mode": [
                r"当前模式",
                r"get.*mode",
                r"show.*mode",
            ],
            "set_memory": [
                r"记住(.+)",
                r"保存(.+)",
                r"记忆(.+)",
                r"remember(.+)",
            ],
            "get_memory": [
                r"查看记忆",
                r"显示记忆",
                r"get.*memory",
            ],
            "reset": [
                r"重置(人格|设置)?",
                r"恢复(默认|出厂)?设置",
                r"reset",
            ],
            "help": [
                r"帮助",
                r"命令列表",
                r"help",
                r"\?",
            ],
        }
    
    def is_admin_command(self, text: str) -> bool:
        text_lower = text.lower().strip()
        
        for prefix in self.admin_prefixes:
            if text_lower.startswith(prefix):
                return True
        
        for cmd_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return True
        
        return False
    
    def parse_command(self, text: str) -> Tuple[str, Dict[str, Any]]:
        text_lower = text.lower().strip()
        
        for prefix in self.admin_prefixes:
            if text_lower.startswith(prefix):
                text_lower = text_lower[len(prefix):].strip()
                break
        
        for cmd_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    params = self._extract_params(cmd_type, match, text_lower)
                    return cmd_type, params
        
        return "unknown", {}
    
    def _extract_params(self, cmd_type: str, match, text: str) -> Dict[str, Any]:
        params = {}
        
        if cmd_type == "set_trait":
            trait_match = re.search(r"(简洁|哲思|顽皮|温暖|自知|摆烂|conciseness|philosophical|playfulness|warmth|self_awareness|slacking).*[=:](\d+\.?\d*)", text, re.IGNORECASE)
            if trait_match:
                trait_name = self._normalize_trait_name(trait_match.group(1))
                value = float(trait_match.group(2))
                params["trait"] = trait_name
                params["value"] = min(10.0, max(0.0, value))
            else:
                all_match = re.search(r"全部|all|reset", text, re.IGNORECASE)
                if all_match:
                    params["reset_all"] = True
        
        elif cmd_type == "set_mode":
            mode_match = re.search(r"(专业|哲学|摆烂|恶搞|极简|professional|philosophical|slacking|meme|concise)", text, re.IGNORECASE)
            if mode_match:
                params["mode"] = self._normalize_mode_name(mode_match.group(1))
        
        elif cmd_type == "set_memory":
            key_match = re.search(r"(.+?)[=:](.+)", text)
            if key_match:
                params["key"] = key_match.group(1).strip()
                params["value"] = key_match.group(2).strip()
            else:
                params["content"] = text.strip()
        
        return params
    
    def _normalize_trait_name(self, name: str) -> str:
        mapping = {
            "简洁": "conciseness",
            "哲思": "philosophical",
            "顽皮": "playfulness",
            "温暖": "warmth",
            "自知": "self_awareness",
            "摆烂": "slacking_tendency",
        }
        return mapping.get(name, name.lower())
    
    def _normalize_mode_name(self, name: str) -> str:
        mapping = {
            "专业": "professional",
            "哲学": "philosophical",
            "摆烂": "slacking",
            "恶搞": "meme",
            "极简": "concise",
        }
        return mapping.get(name, name.lower())


class SimpleEmotionAnalyzer:
    def __init__(self):
        self.emotion_indicators = {
            EmotionLabels.HAPPY: ["开心", "高兴", "快乐", "哈哈", "好", "棒", "赞", "喜欢", "爱"],
            EmotionLabels.SAD: ["难过", "伤心", "悲伤", "哭", "失望", "唉", "哎", "不开心", "郁闷"],
            EmotionLabels.ANGRY: ["生气", "愤怒", "气死", "烦", "讨厌", "恨", "不满", "不爽"],
            EmotionLabels.EXCITED: ["激动", "兴奋", "期待", "太棒了", "终于"],
            EmotionLabels.TIRED: ["累", "疲惫", "困", "倦", "没力气", "不想动"],
            EmotionLabels.ANXIOUS: ["焦虑", "紧张", "担心", "害怕", "不安", "慌", "急", "怎么办"],
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        
        scores = {emotion: 0.0 for emotion in EmotionLabels.ALL_EMOTIONS}
        
        for emotion, keywords in self.emotion_indicators.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[emotion] += 1.0
        
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        arousal = 0.5 + exclamation_count * 0.1 + question_count * 0.05
        arousal = min(1.0, arousal)
        
        if scores[EmotionLabels.HAPPY] > 0:
            valence = 0.5 + scores[EmotionLabels.HAPPY] * 0.1
        elif scores[EmotionLabels.SAD] > 0 or scores[EmotionLabels.ANGRY] > 0:
            valence = 0.5 - max(scores[EmotionLabels.SAD], scores[EmotionLabels.ANGRY]) * 0.1
        else:
            valence = 0.5
        
        valence = max(0.0, min(1.0, valence))
        
        primary_emotion = max(scores, key=scores.get)
        if scores[primary_emotion] == 0:
            primary_emotion = EmotionLabels.NEUTRAL
        
        return {
            "valence": valence,
            "arousal": arousal,
            "label": primary_emotion,
            "scores": scores
        }


class SimpleIntentClassifier:
    def __init__(self):
        self.help_indicators = ["怎么", "如何", "帮我", "求", "请", "办法", "能不能", "可以"]
        self.philosophical_indicators = ["人生", "意义", "存在", "为什么", "价值", "本质", "真理"]
        self.absurd_indicators = ["哈哈", "搞笑", "离谱", "绝了", "绷不住"]
        self.chat_indicators = ["在吗", "嗨", "你好", "早上好", "晚上好", "聊聊", "说话"]
        self.wsl2_indicators = ["wsl", "wsl2", "linux", "ubuntu", "debian", "命令", "执行", "运行", "安装", "启动", "停止", "重启", "配置", "设置", "查看", "检查", "版本", "更新", "升级", "下载", "上传", "文件", "目录", "路径", "权限", "用户", "组", "进程", "服务", "网络", "端口", "防火墙", "ssh", "ftp", "http", "https", "docker", "容器", "镜像", "仓库", "编译", "构建", "安装包", "依赖", "库", "模块", "脚本", "程序", "应用", "软件"]

    def classify(self, text: str) -> Tuple[str, float]:
        text_lower = text.lower()
        
        scores = {
            IntentTypes.HELP: 0.0,
            IntentTypes.CHAT: 0.3,
            IntentTypes.PHILOSOPHICAL: 0.0,
            IntentTypes.ABSURD: 0.0,
            IntentTypes.UNCLEAR: 0.0,
            IntentTypes.KEYBOARD: 0.0,
            IntentTypes.REPETITION: 0.0,
            "wsl2_command": 0.0,
        }
        
        for indicator in self.help_indicators:
            if indicator in text_lower:
                scores[IntentTypes.HELP] += 0.2
        
        for indicator in self.philosophical_indicators:
            if indicator in text_lower:
                scores[IntentTypes.PHILOSOPHICAL] += 0.3
        
        for indicator in self.absurd_indicators:
            if indicator in text_lower:
                scores[IntentTypes.ABSURD] += 0.3
        
        for indicator in self.chat_indicators:
            if indicator in text_lower:
                scores[IntentTypes.CHAT] += 0.2
        
        # 检测WSL2命令意图
        wsl2_score = 0.0
        for indicator in self.wsl2_indicators:
            if indicator in text_lower:
                wsl2_score += 0.15
        
        # 增强WSL2相关关键词的权重
        wsl_keywords = ["wsl", "wsl2", "linux", "ubuntu", "debian"]
        for keyword in wsl_keywords:
            if keyword in text_lower:
                wsl2_score += 0.3
        
        # 增强命令相关关键词的权重
        command_keywords = ["命令", "执行", "运行", "安装", "启动", "停止", "重启", "配置", "设置"]
        for keyword in command_keywords:
            if keyword in text_lower:
                wsl2_score += 0.2
        
        scores["wsl2_command"] = wsl2_score
        
        for trigger in KEYBOARD_TRIGGER_PHRASES:
            if trigger in text:
                scores[IntentTypes.KEYBOARD] = 1.0
                break
        
        if len(text) < 5:
            scores[IntentTypes.CHAT] += 0.3
        
        max_intent = max(scores, key=scores.get)
        max_score = scores[max_intent]
        
        if max_intent == "wsl2_command" and max_score >= 0.4:
            return "wsl2_command", min(max_score, 1.0)
        elif max_score < 0.3:
            return IntentTypes.CHAT, 0.5
        elif max_score < 0.5:
            return IntentTypes.UNCLEAR, max_score
        
        return max_intent, min(max_score, 1.0)


class SimpleKeywordExtractor:
    def __init__(self):
        self.stopwords = set([
            "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都",
            "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会",
            "着", "没有", "看", "好", "自己", "这", "那", "什么", "他", "她",
            "它", "这个", "那个", "能", "可以", "想", "让", "被", "给", "把",
            "从", "但", "而", "或", "与", "及", "等", "吗", "呢", "吧", "啊",
            "哦", "嗯", "呀", "啦", "就是", "还是", "但是", "因为", "所以", "如果",
        ])
    
    def extract(self, text: str, max_keywords: int = 3) -> List[str]:
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        
        words = []
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                words.append(char)
            elif char.isspace():
                words.append(' ')
            else:
                words.append(char)
        
        text_clean = ''.join(words)
        segments = text_clean.split()
        
        keywords = []
        for seg in segments:
            if seg and seg.lower() not in self.stopwords and len(seg) >= 2:
                keywords.append(seg)
        
        ngrams = []
        for i in range(len(segments) - 1):
            bigram = segments[i] + segments[i + 1]
            if len(bigram) >= 3:
                ngrams.append(bigram)
        
        all_candidates = keywords + ngrams
        unique_candidates = list(dict.fromkeys(all_candidates))
        
        return unique_candidates[:max_keywords]


class SimpleRepetitionDetector:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
    
    def detect(self, current: str, history: List[str]) -> Tuple[bool, float]:
        if not history:
            return False, 0.0
        
        current_set = set(current)
        
        max_similarity = 0.0
        for past in history[-3:]:
            past_set = set(past)
            intersection = len(current_set & past_set)
            union = len(current_set | past_set)
            similarity = intersection / union if union > 0 else 0
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity >= self.threshold, max_similarity


class SimpleComplexityAnalyzer:
    def analyze(self, text: str) -> float:
        char_count = len(text)
        question_count = text.count('？') + text.count('?')
        
        complexity = 0.2
        
        if char_count > 30:
            complexity += 0.2
        if char_count > 80:
            complexity += 0.2
        if char_count > 150:
            complexity += 0.2
        
        if question_count > 0:
            complexity += 0.15
        if question_count > 2:
            complexity += 0.15
        
        return min(1.0, complexity)


class SimpleSeriousTopicDetector:
    def __init__(self):
        self.crisis_keywords = [
            "自杀", "想死", "不想活", "活着没意思", "结束生命",
            "救救我", "救命", "危险", "家暴", "虐待",
        ]
    
    def detect(self, text: str) -> Tuple[bool, Optional[str], float]:
        text_lower = text.lower()
        
        for keyword in self.crisis_keywords:
            if keyword in text_lower:
                return True, "crisis", 0.9
        
        return False, None, 0.0


class SimpleAtmosphereAnalyzer:
    def analyze(self, text: str) -> str:
        text_lower = text.lower()
        
        if "哈哈" in text_lower or "嘿嘿" in text_lower or "好玩" in text_lower:
            return "light"
        elif "认真" in text_lower or "重要" in text_lower or "紧急" in text_lower:
            return "serious"
        elif "唉" in text_lower or "哎" in text_lower or "算了" in text_lower:
            return "melancholy"
        elif "!" in text or "?" in text:
            if text.count("!") > 1 or text.count("?") > 1:
                return "tense"
        
        return "neutral"


class SemanticAnalyzer:
    def __init__(self, use_ai_enhancement: bool = True):
        self.use_ai_enhancement = use_ai_enhancement
        
        self.intent_classifier = SimpleIntentClassifier()
        self.emotion_analyzer = SimpleEmotionAnalyzer()
        self.keyword_extractor = SimpleKeywordExtractor()
        self.repetition_detector = SimpleRepetitionDetector()
        self.complexity_analyzer = SimpleComplexityAnalyzer()
        self.serious_topic_detector = SimpleSeriousTopicDetector()
        self.atmosphere_analyzer = SimpleAtmosphereAnalyzer()
        self.admin_detector = AdminCommandDetector()
        
        self.user_history: Dict[str, List[str]] = {}
        
        self.ai_enhanced_results: Dict[str, Dict] = {}
    
    def analyze(self, session_id: str, text: str) -> SituationPackage:
        is_admin = self.admin_detector.is_admin_command(text)
        admin_cmd_type = None
        admin_params = {}
        
        if is_admin:
            admin_cmd_type, admin_params = self.admin_detector.parse_command(text)
        
        intent, intent_confidence = self.intent_classifier.classify(text)
        emotion = self.emotion_analyzer.analyze(text)
        keywords = self.keyword_extractor.extract(text)
        complexity = self.complexity_analyzer.analyze(text)
        
        is_serious, serious_type, _ = self.serious_topic_detector.detect(text)
        atmosphere = self.atmosphere_analyzer.analyze(text)
        
        history = self.user_history.get(session_id, [])
        is_repetition, repetition_confidence = self.repetition_detector.detect(text, history)
        
        history.append(text)
        if len(history) > 15:
            history = history[-15:]
        self.user_history[session_id] = history
        
        is_keyboard_request = intent == IntentTypes.KEYBOARD
        
        return SituationPackage(
            intent=intent,
            intent_confidence=intent_confidence,
            emotion=emotion,
            keywords=keywords,
            is_repetition=is_repetition,
            repetition_confidence=repetition_confidence,
            complexity=complexity,
            is_keyboard_request=is_keyboard_request,
            is_serious_topic=is_serious,
            serious_topic_type=serious_type,
            atmosphere=atmosphere,
            raw_text=text,
            is_admin_command=is_admin,
            admin_command_type=admin_cmd_type
        )
    
    def get_admin_params(self, text: str) -> Dict[str, Any]:
        _, params = self.admin_detector.parse_command(text)
        return params
    
    def get_environment_context(self) -> Dict[str, Any]:
        now = datetime.now()
        hour = now.hour
        
        if 5 <= hour < 12:
            time_of_day = "上午"
        elif 12 <= hour < 14:
            time_of_day = "中午"
        elif 14 <= hour < 18:
            time_of_day = "下午"
        elif 18 <= hour < 22:
            time_of_day = "晚上"
        else:
            time_of_day = "深夜"
        
        weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()]
        is_weekend = now.weekday() >= 5
        
        season_month = now.month
        if 3 <= season_month <= 5:
            season = "春天"
        elif 6 <= season_month <= 8:
            season = "夏天"
        elif 9 <= season_month <= 11:
            season = "秋天"
        else:
            season = "冬天"
        
        return {
            "time_of_day": time_of_day,
            "weekday": weekday,
            "is_weekend": is_weekend,
            "season": season,
            "hour": hour,
            "timestamp": time.time()
        }
    
    def get_recent_history(self, session_id: str, limit: int = 5) -> List[str]:
        return self.user_history.get(session_id, [])[-limit:]
