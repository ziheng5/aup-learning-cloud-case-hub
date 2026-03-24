import random
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from config import StateModes, IntentTypes, EmotionLabels, PersonalityConfig


@dataclass
class StateDecision:
    mode: str
    confidence: float
    reasoning: str
    allow_override: bool = True
    is_locked: bool = False
    lock_reason: str = ""


class StateDecisionEngine:
    def __init__(self):
        self.base_probabilities = {
            StateModes.PROFESSIONAL: 0.20,
            StateModes.PHILOSOPHICAL: 0.20,
            StateModes.SLACKING: 0.15,
            StateModes.MEME: 0.20,
            StateModes.CONCISE: 0.25,
        }
        
        self.intent_weights = {
            IntentTypes.HELP: {
                StateModes.PROFESSIONAL: 2.5,
                StateModes.PHILOSOPHICAL: 0.3,
                StateModes.SLACKING: 0.05,
                StateModes.MEME: 0.05,
                StateModes.CONCISE: 0.8,
            },
            IntentTypes.PHILOSOPHICAL: {
                StateModes.PROFESSIONAL: 0.4,
                StateModes.PHILOSOPHICAL: 3.0,
                StateModes.SLACKING: 0.2,
                StateModes.MEME: 0.1,
                StateModes.CONCISE: 0.3,
            },
            IntentTypes.ABSURD: {
                StateModes.PROFESSIONAL: 0.1,
                StateModes.PHILOSOPHICAL: 0.2,
                StateModes.SLACKING: 1.5,
                StateModes.MEME: 4.0,
                StateModes.CONCISE: 0.3,
            },
            IntentTypes.CHAT: {
                StateModes.PROFESSIONAL: 0.6,
                StateModes.PHILOSOPHICAL: 1.2,
                StateModes.SLACKING: 1.0,
                StateModes.MEME: 1.5,
                StateModes.CONCISE: 1.0,
            },
            IntentTypes.UNCLEAR: {
                StateModes.PROFESSIONAL: 0.8,
                StateModes.PHILOSOPHICAL: 0.6,
                StateModes.SLACKING: 1.0,
                StateModes.MEME: 0.6,
                StateModes.CONCISE: 2.0,
            },
            IntentTypes.REPETITION: {
                StateModes.PROFESSIONAL: 0.3,
                StateModes.PHILOSOPHICAL: 0.2,
                StateModes.SLACKING: 3.0,
                StateModes.MEME: 0.8,
                StateModes.CONCISE: 1.5,
            },
            IntentTypes.KEYBOARD: {
                StateModes.PROFESSIONAL: 0.5,
                StateModes.PHILOSOPHICAL: 0.05,
                StateModes.SLACKING: 0.05,
                StateModes.MEME: 0.05,
                StateModes.CONCISE: 4.0,
            },
        }
        
        self.emotion_weights = {
            EmotionLabels.HAPPY: {
                StateModes.MEME: 1.5,
                StateModes.PHILOSOPHICAL: 0.6,
                StateModes.PROFESSIONAL: 0.8,
            },
            EmotionLabels.SAD: {
                StateModes.PROFESSIONAL: 1.5,
                StateModes.MEME: 0.1,
                StateModes.SLACKING: 0.3,
                StateModes.PHILOSOPHICAL: 1.2,
            },
            EmotionLabels.ANGRY: {
                StateModes.PROFESSIONAL: 2.0,
                StateModes.MEME: 0.05,
                StateModes.SLACKING: 0.05,
            },
            EmotionLabels.TIRED: {
                StateModes.SLACKING: 2.0,
                StateModes.PHILOSOPHICAL: 1.5,
                StateModes.MEME: 0.5,
                StateModes.CONCISE: 1.2,
            },
            EmotionLabels.EXCITED: {
                StateModes.MEME: 2.0,
                StateModes.PHILOSOPHICAL: 0.5,
                StateModes.PROFESSIONAL: 0.6,
            },
            EmotionLabels.ANXIOUS: {
                StateModes.PROFESSIONAL: 1.8,
                StateModes.SLACKING: 0.2,
                StateModes.MEME: 0.3,
            },
        }
        
        self.atmosphere_weights = {
            "light": {
                StateModes.MEME: 1.5,
                StateModes.SLACKING: 1.2,
            },
            "serious": {
                StateModes.PROFESSIONAL: 2.0,
                StateModes.MEME: 0.3,
                StateModes.SLACKING: 0.3,
            },
            "melancholy": {
                StateModes.PHILOSOPHICAL: 1.8,
                StateModes.SLACKING: 1.3,
                StateModes.MEME: 0.4,
            },
            "tense": {
                StateModes.PROFESSIONAL: 1.5,
                StateModes.CONCISE: 1.3,
                StateModes.MEME: 0.2,
            },
        }
        
        self.time_weights = {
            "深夜": {
                StateModes.PHILOSOPHICAL: 1.8,
                StateModes.SLACKING: 1.5,
                StateModes.PROFESSIONAL: 0.5,
            },
            "晚上": {
                StateModes.PHILOSOPHICAL: 1.3,
                StateModes.MEME: 1.2,
                StateModes.SLACKING: 1.1,
            },
            "上午": {
                StateModes.PROFESSIONAL: 1.2,
                StateModes.CONCISE: 1.1,
            },
        }
        
        self.weekend_weights = {
            True: {
                StateModes.MEME: 1.3,
                StateModes.PHILOSOPHICAL: 1.2,
                StateModes.SLACKING: 1.2,
                StateModes.PROFESSIONAL: 0.7,
            },
            False: {
                StateModes.PROFESSIONAL: 1.1,
                StateModes.CONCISE: 1.1,
            },
        }

    def _calculate_situation_match(self, situation: Any) -> Dict[str, float]:
        match_scores = {mode: 0.0 for mode in StateModes.ALL_MODES}
        
        if situation.intent in self.intent_weights:
            for mode, weight in self.intent_weights[situation.intent].items():
                match_scores[mode] += weight * 2.0
        
        emotion_label = situation.emotion.get("label", EmotionLabels.NEUTRAL)
        if emotion_label in self.emotion_weights:
            for mode, weight in self.emotion_weights[emotion_label].items():
                match_scores[mode] += weight
        
        return match_scores

    def _apply_personality_bias(self, probabilities: Dict[str, float], 
                                  personality: Dict[str, float]) -> Dict[str, float]:
        weighted = probabilities.copy()
        
        philosophical = personality.get("philosophical", 5.0)
        playfulness = personality.get("playfulness", 5.0)
        slacking = personality.get("slacking_tendency", 5.0)
        conciseness = personality.get("conciseness", 5.0)
        
        if philosophical > 7.0:
            weighted[StateModes.PHILOSOPHICAL] *= 1.0 + (philosophical - 7.0) * 0.1
        if playfulness > 6.0:
            weighted[StateModes.MEME] *= 1.0 + (playfulness - 6.0) * 0.1
        if slacking > 6.0:
            weighted[StateModes.SLACKING] *= 1.0 + (slacking - 6.0) * 0.1
        if conciseness > 7.0:
            weighted[StateModes.CONCISE] *= 1.0 + (conciseness - 7.0) * 0.05
        
        return weighted

    def _apply_environment_bias(self, probabilities: Dict[str, float],
                                 environment: Dict[str, Any]) -> Dict[str, float]:
        weighted = probabilities.copy()
        
        time_of_day = environment.get("time_of_day", "")
        if time_of_day in self.time_weights:
            for mode, weight in self.time_weights[time_of_day].items():
                weighted[mode] *= weight
        
        is_weekend = environment.get("is_weekend", False)
        if is_weekend in self.weekend_weights:
            for mode, weight in self.weekend_weights[is_weekend].items():
                weighted[mode] *= weight
        
        return weighted

    def _apply_atmosphere_bias(self, probabilities: Dict[str, float],
                                 atmosphere: str) -> Dict[str, float]:
        weighted = probabilities.copy()
        
        if atmosphere in self.atmosphere_weights:
            for mode, weight in self.atmosphere_weights[atmosphere].items():
                weighted[mode] *= weight
        
        return weighted

    def _apply_repetition_bias(self, probabilities: Dict[str, float],
                                is_repetition: bool,
                                repetition_confidence: float) -> Dict[str, float]:
        if not is_repetition:
            return probabilities
        
        weighted = probabilities.copy()
        factor = 1.0 + repetition_confidence
        
        weighted[StateModes.SLACKING] *= factor
        weighted[StateModes.CONCISE] *= (1.0 + repetition_confidence * 0.5)
        weighted[StateModes.PROFESSIONAL] *= max(0.3, 1.0 - repetition_confidence * 0.5)
        weighted[StateModes.MEME] *= max(0.5, 1.0 - repetition_confidence * 0.3)
        
        return weighted

    def _add_random_noise(self, probabilities: Dict[str, float], 
                          noise_level: float = 0.1) -> Dict[str, float]:
        noisy = {}
        for mode, prob in probabilities.items():
            noise = random.uniform(1 - noise_level, 1 + noise_level)
            noisy[mode] = prob * noise
        return noisy

    def _normalize(self, probabilities: Dict[str, float]) -> Dict[str, float]:
        total = sum(probabilities.values())
        if total == 0:
            return {mode: 1.0 / len(probabilities) for mode in probabilities}
        return {mode: prob / total for mode, prob in probabilities.items()}

    def _check_serious_lock(self, situation: Any) -> Tuple[bool, str]:
        if situation.is_serious_topic:
            topic_type = situation.serious_topic_type
            if topic_type in ["medical", "crisis", "safety"]:
                return True, f"严肃话题锁定: {topic_type}"
        
        emotion_label = situation.emotion.get("label", "")
        if emotion_label == EmotionLabels.ANGRY:
            return True, "用户情绪激动，需要认真对待"
        
        if situation.complexity > 0.8:
            return True, "问题复杂度高，需要认真回答"
        
        return False, ""

    def decide(self, situation: Any, personality: Dict[str, float],
               environment: Dict[str, Any]) -> StateDecision:
        
        if situation.is_keyboard_request:
            return StateDecision(
                mode=StateModes.CONCISE,
                confidence=1.0,
                reasoning="键盘控制请求，使用简洁模式",
                allow_override=False,
                is_locked=True,
                lock_reason="键盘控制模式锁定"
            )
        
        is_locked, lock_reason = self._check_serious_lock(situation)
        if is_locked:
            return StateDecision(
                mode=StateModes.PROFESSIONAL,
                confidence=0.95,
                reasoning=lock_reason,
                allow_override=False,
                is_locked=True,
                lock_reason=lock_reason
            )
        
        probabilities = self.base_probabilities.copy()
        
        situation_match = self._calculate_situation_match(situation)
        for mode in probabilities:
            probabilities[mode] *= situation_match.get(mode, 1.0)
        
        probabilities = self._apply_personality_bias(probabilities, personality)
        
        probabilities = self._apply_environment_bias(probabilities, environment)
        
        probabilities = self._apply_atmosphere_bias(probabilities, situation.atmosphere)
        
        probabilities = self._apply_repetition_bias(
            probabilities, situation.is_repetition, situation.repetition_confidence
        )
        
        noise_level = 0.1 + (1 - situation.intent_confidence) * 0.1
        probabilities = self._add_random_noise(probabilities, noise_level)
        
        probabilities = self._normalize(probabilities)
        
        modes = list(probabilities.keys())
        weights = list(probabilities.values())
        chosen_mode = random.choices(modes, weights=weights, k=1)[0]
        confidence = probabilities[chosen_mode]
        
        reasoning = self._generate_reasoning(situation, chosen_mode, environment)
        
        return StateDecision(
            mode=chosen_mode,
            confidence=confidence,
            reasoning=reasoning,
            is_locked=False
        )

    def _generate_reasoning(self, situation: Any, mode: str,
                            environment: Dict[str, Any]) -> str:
        intent_desc = {
            IntentTypes.HELP: "求助类问题",
            IntentTypes.PHILOSOPHICAL: "哲学性话题",
            IntentTypes.ABSURD: "荒诞/玩梗",
            IntentTypes.CHAT: "闲聊",
            IntentTypes.UNCLEAR: "表达不清",
            IntentTypes.REPETITION: "重复问题",
            IntentTypes.KEYBOARD: "键盘控制",
        }
        
        mode_desc = {
            StateModes.PROFESSIONAL: "专业模式",
            StateModes.PHILOSOPHICAL: "哲思模式",
            StateModes.SLACKING: "摆烂模式",
            StateModes.MEME: "恶搞模式",
            StateModes.CONCISE: "极简模式",
        }
        
        parts = []
        parts.append(f"意图: {intent_desc.get(situation.intent, situation.intent)}")
        parts.append(f"氛围: {situation.atmosphere}")
        
        if situation.is_repetition:
            parts.append("(重复问题)")
        
        parts.append(f"→ {mode_desc.get(mode, mode)}")
        
        return " ".join(parts)


class PersonalityCore:
    def __init__(self, data_path: str = "./data/personality.json"):
        self.data_path = Path(data_path)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._load_or_init()
        self._init_history()

    def _load_or_init(self) -> None:
        if self.data_path.exists():
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.traits = data.get("traits", PersonalityConfig.BASE_TRAITS.copy())
                self.mood = data.get("mood", {"current": "neutral", "persistence": 0.7})
                self.habits = data.get("habits", [])
                self.mode_history = data.get("mode_history", [])
        else:
            self.traits = PersonalityConfig.BASE_TRAITS.copy()
            self.mood = {"current": "neutral", "persistence": 0.7}
            self.habits = []
            self.mode_history = []
            self._save()

    def _init_history(self) -> None:
        if not hasattr(self, 'mode_history'):
            self.mode_history = []

    def _save(self) -> None:
        data = {
            "traits": self.traits,
            "mood": self.mood,
            "habits": self.habits,
            "mode_history": self.mode_history[-100:],
            "last_updated": datetime.now().isoformat()
        }
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_traits(self) -> Dict[str, float]:
        return self.traits.copy()

    def get_mood(self) -> str:
        return self.mood.get("current", "neutral")

    def update_trait(self, trait_name: str, delta: float) -> None:
        if trait_name not in self.traits:
            return
        
        current = self.traits[trait_name]
        new_value = current + delta
        new_value = max(PersonalityConfig.TRAIT_MIN, min(PersonalityConfig.TRAIT_MAX, new_value))
        self.traits[trait_name] = new_value
        self._save()

    def update_mood(self, new_mood: str) -> None:
        persistence = self.mood.get("persistence", 0.7)
        if random.random() < persistence:
            return
        self.mood["current"] = new_mood
        self._save()

    def record_mode_usage(self, mode: str, user_satisfaction: float = None) -> None:
        record = {
            "mode": mode,
            "timestamp": time.time(),
            "user_satisfaction": user_satisfaction
        }
        self.mode_history.append(record)
        if len(self.mode_history) > 100:
            self.mode_history = self.mode_history[-100:]
        self._save()

    def get_mode_statistics(self) -> Dict[str, Dict[str, Any]]:
        stats = {}
        for record in self.mode_history:
            mode = record["mode"]
            if mode not in stats:
                stats[mode] = {"count": 0, "total_satisfaction": 0.0}
            stats[mode]["count"] += 1
            if record.get("user_satisfaction") is not None:
                stats[mode]["total_satisfaction"] += record["user_satisfaction"]
        
        for mode in stats:
            if stats[mode]["count"] > 0:
                stats[mode]["avg_satisfaction"] = stats[mode]["total_satisfaction"] / stats[mode]["count"]
        
        return stats

    def add_habit(self, trigger: str, behavior: str, effectiveness: float = 0.5) -> None:
        existing = next((h for h in self.habits if h["trigger"] == trigger), None)
        if existing:
            existing["count"] = existing.get("count", 0) + 1
            existing["effectiveness"] = (existing.get("effectiveness", 0.5) + effectiveness) / 2
        else:
            self.habits.append({
                "trigger": trigger,
                "behavior": behavior,
                "count": 1,
                "effectiveness": effectiveness,
                "created_at": datetime.now().isoformat()
            })
        
        self.habits = sorted(
            self.habits,
            key=lambda x: (x.get("count", 0), x.get("effectiveness", 0)),
            reverse=True
        )[:20]
        self._save()

    def format_for_prompt(self) -> str:
        lines = []
        lines.append(f"你的名字是：{PersonalityConfig.PET_NAME}")
        lines.append(f"你的身份是：{PersonalityConfig.PET_IDENTITY}")
        lines.append("")
        lines.append("当前你的人格参数：")
        
        trait_names = {
            "conciseness": "简洁度",
            "philosophical": "哲思倾向",
            "playfulness": "顽皮度",
            "warmth": "温暖度",
            "self_awareness": "自知之明",
            "slacking_tendency": "摆烂倾向"
        }
        
        for trait, value in self.traits.items():
            name = trait_names.get(trait, trait)
            lines.append(f"- {name}: {value:.1f}/10")
        
        lines.append("")
        lines.append(f"当前情绪状态：{self.mood.get('current', 'neutral')}")
        
        if self.habits:
            lines.append("")
            lines.append("你的一些习惯：")
            for habit in self.habits[:3]:
                lines.append(f"- 当{habit['trigger']}时，倾向于{habit['behavior']}")
        
        return "\n".join(lines)
