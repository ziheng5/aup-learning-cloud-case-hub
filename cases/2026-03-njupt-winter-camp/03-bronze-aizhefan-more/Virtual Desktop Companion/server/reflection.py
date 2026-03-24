import asyncio
import json
import time
import random
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

from memory import MemorySystem
from state_machine import PersonalityCore, StateDecisionEngine
from config import settings, StateModes, EmotionLabels


@dataclass
class ReflectionResult:
    memory_updates: List[Dict[str, Any]]
    personality_adjustments: List[Dict[str, Any]]
    new_habits: List[Dict[str, Any]]
    insights: List[str]
    mode_feedback: Dict[str, float]


@dataclass
class BehaviorFeedback:
    mode: str
    user_reaction: str
    response_time: float
    engagement_level: float
    timestamp: float


@dataclass
class UserFeedbackSignal:
    type: str
    intensity: float
    context: str
    timestamp: float


class FeedbackAnalyzer:
    def __init__(self):
        self.positive_patterns = [
            r"哈哈", r"笑死", r"太对了", r"好棒", r"厉害", r"666", r"爱了",
            r"谢谢", r"感谢", r"太好", r"完美", r"正合我意", r"说到点上",
            r"有道理", r"确实", r"没错", r"继续", r"再来",
        ]
        self.negative_patterns = [
            r"不对", r"不是", r"错了", r"不好", r"不行", r"别这样",
            r"太啰嗦", r"太长了", r"看不懂", r"什么意思", r"说人话",
            r"敷衍", r"太随意", r"认真点", r"严肃",
        ]
        self.neutral_patterns = [
            r"然后", r"继续说", r"还有", r"嗯", r"哦", r"好的",
        ]
    
    def analyze_text_feedback(self, text: str) -> Tuple[str, float]:
        text_lower = text.lower()
        
        positive_score = 0
        for pattern in self.positive_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                positive_score += 1
        
        negative_score = 0
        for pattern in self.negative_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                negative_score += 1
        
        if positive_score > negative_score:
            return "positive", min(1.0, positive_score * 0.3)
        elif negative_score > positive_score:
            return "negative", min(1.0, negative_score * 0.3)
        else:
            return "neutral", 0.0
    
    def analyze_response_time(self, response_time: float) -> Tuple[str, float]:
        if response_time < 1.0:
            return "fast", 0.8
        elif response_time < 3.0:
            return "normal", 0.5
        elif response_time < 5.0:
            return "slow", 0.3
        else:
            return "too_slow", 0.1
    
    def analyze_message_length(self, user_msg: str, ai_msg: str) -> Tuple[str, float]:
        user_len = len(user_msg)
        ai_len = len(ai_msg)
        
        ratio = ai_len / max(user_len, 1)
        
        if ratio > 5.0 and user_len < 20:
            return "too_verbose", -0.3
        elif ratio > 10.0:
            return "too_verbose", -0.2
        elif ratio < 0.3 and user_len > 20:
            return "too_concise", -0.1
        else:
            return "appropriate", 0.2


class ReflectionEngine:
    def __init__(self, memory: MemorySystem, personality: PersonalityCore,
                 decision_engine: Optional[StateDecisionEngine] = None):
        self.memory = memory
        self.personality = personality
        self.decision_engine = decision_engine
        self.log_path = Path("./data/reflection_logs")
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        self.interaction_buffer: List[Dict[str, Any]] = []
        self.behavior_feedback_buffer: List[BehaviorFeedback] = []
        self.last_reflection_time = 0
        
        self.mode_performance = defaultdict(lambda: {"success": 0, "failure": 0, "total": 0})
        self.trait_correlations = defaultdict(lambda: defaultdict(float))
        
        self.feedback_analyzer = FeedbackAnalyzer()
        self.weight_adjuster = WeightAdjustmentEngine(decision_engine) if decision_engine else None
        
        self.feedback_history: List[Dict[str, Any]] = []
        self.learning_rate = 0.05
        self.momentum = 0.9
    
    def log_interaction(self, interaction: Dict[str, Any]) -> None:
        self.interaction_buffer.append({
            **interaction,
            "timestamp": time.time()
        })
    
    def log_behavior_feedback(self, feedback: BehaviorFeedback) -> None:
        self.behavior_feedback_buffer.append(feedback)
        
        mode = feedback.mode
        if feedback.engagement_level > 0.6:
            self.mode_performance[mode]["success"] += 1
        elif feedback.engagement_level < 0.3:
            self.mode_performance[mode]["failure"] += 1
        self.mode_performance[mode]["total"] += 1
    
    def process_immediate_feedback(self, user_message: str, ai_response: str,
                                   mode: str, intent: str, emotion: str,
                                   response_time: float) -> Dict[str, Any]:
        feedback_type, feedback_intensity = self.feedback_analyzer.analyze_text_feedback(user_message)
        speed_feedback, speed_score = self.feedback_analyzer.analyze_response_time(response_time)
        length_feedback, length_score = self.feedback_analyzer.analyze_message_length(user_message, ai_response)
        
        overall_score = (feedback_intensity + speed_score + length_score) / 3
        
        feedback_record = {
            "timestamp": time.time(),
            "mode": mode,
            "intent": intent,
            "emotion": emotion,
            "text_feedback": feedback_type,
            "text_intensity": feedback_intensity,
            "speed_feedback": speed_feedback,
            "speed_score": speed_score,
            "length_feedback": length_feedback,
            "length_score": length_score,
            "overall_score": overall_score
        }
        
        self.feedback_history.append(feedback_record)
        if len(self.feedback_history) > 1000:
            self.feedback_history = self.feedback_history[-1000:]
        
        if self.weight_adjuster:
            reward = overall_score - 0.5
            if abs(reward) > 0.1:
                self.weight_adjuster.adjust_intent_weights(intent, mode, reward)
                self.weight_adjuster.adjust_emotion_weights(emotion, mode, reward * 0.5)
        
        if feedback_type == "positive":
            self.mode_performance[mode]["success"] += 1
        elif feedback_type == "negative":
            self.mode_performance[mode]["failure"] += 1
        self.mode_performance[mode]["total"] += 1
        
        return feedback_record
    
    def get_feedback_statistics(self, window_minutes: int = 30) -> Dict[str, Any]:
        cutoff = time.time() - window_minutes * 60
        recent = [f for f in self.feedback_history if f.get("timestamp", 0) >= cutoff]
        
        if not recent:
            return {"error": "没有足够的反馈数据"}
        
        mode_stats = defaultdict(lambda: {"count": 0, "avg_score": 0.0})
        intent_stats = defaultdict(lambda: {"count": 0, "avg_score": 0.0})
        
        for f in recent:
            mode = f["mode"]
            mode_stats[mode]["count"] += 1
            mode_stats[mode]["avg_score"] += f["overall_score"]
            
            intent = f["intent"]
            intent_stats[intent]["count"] += 1
            intent_stats[intent]["avg_score"] += f["overall_score"]
        
        for stats in mode_stats.values():
            if stats["count"] > 0:
                stats["avg_score"] /= stats["count"]
        
        for stats in intent_stats.values():
            if stats["count"] > 0:
                stats["avg_score"] /= stats["count"]
        
        return {
            "total_feedbacks": len(recent),
            "window_minutes": window_minutes,
            "mode_performance": dict(mode_stats),
            "intent_performance": dict(intent_stats),
            "overall_avg_score": sum(f["overall_score"] for f in recent) / len(recent)
        }
    
    def apply_feedback_based_adjustments(self) -> Dict[str, Any]:
        stats = self.get_feedback_statistics(window_minutes=60)
        if "error" in stats:
            return {"status": "no_data"}
        
        adjustments = {}
        
        for mode, mode_stats in stats.get("mode_performance", {}).items():
            if mode_stats["count"] < 3:
                continue
            
            avg_score = mode_stats["avg_score"]
            trait = self._mode_to_trait(mode)
            
            if avg_score > 0.6:
                adjustments[trait] = adjustments.get(trait, 0) + 0.02
            elif avg_score < 0.3:
                adjustments[trait] = adjustments.get(trait, 0) - 0.02
        
        if adjustments:
            for trait, delta in adjustments.items():
                self.personality.update_trait(trait, delta)
        
        return {
            "status": "applied",
            "adjustments": adjustments,
            "stats": stats
        }
    
    def get_mode_success_rate(self, mode: str) -> float:
        perf = self.mode_performance[mode]
        if perf["total"] == 0:
            return 0.5
        return perf["success"] / perf["total"]
    
    async def run_reflection(self, force: bool = False) -> Optional[ReflectionResult]:
        current_time = time.time()
        time_since_last = current_time - self.last_reflection_time
        
        if not force and time_since_last < settings.REFLECTION_INTERVAL_SECONDS:
            return None
        
        if not self.interaction_buffer and not self.behavior_feedback_buffer:
            return None
        
        interactions = self.interaction_buffer.copy()
        feedbacks = self.behavior_feedback_buffer.copy()
        
        self.interaction_buffer = []
        self.behavior_feedback_buffer = []
        self.last_reflection_time = current_time
        
        result = await self._analyze_interactions(interactions, feedbacks)
        await self._apply_reflection(result)
        await self._save_reflection_log(interactions, result)
        
        return result
    
    async def _analyze_interactions(self, interactions: List[Dict[str, Any]],
                                     feedbacks: List[BehaviorFeedback]) -> ReflectionResult:
        memory_updates = []
        personality_adjustments = []
        new_habits = []
        insights = []
        mode_feedback = defaultdict(float)
        
        topic_freq = defaultdict(int)
        emotion_trends = []
        mode_success = defaultdict(lambda: {"success": 0, "failure": 0})
        time_patterns = defaultdict(list)
        
        for interaction in interactions:
            keywords = interaction.get("keywords", [])
            for kw in keywords:
                topic_freq[kw] += 1
            
            emotion = interaction.get("emotion", "neutral")
            emotion_trends.append(emotion)
            
            mode = interaction.get("mode", "concise")
            hour = datetime.fromtimestamp(interaction.get("timestamp", time.time())).hour
            time_patterns[hour].append(mode)
            
            importance = interaction.get("importance", 0.5)
            if importance > 0.7:
                memory_updates.append({
                    "type": "important",
                    "content": interaction.get("user_input", ""),
                    "importance": importance,
                    "emotion": emotion
                })
            
            user_satisfaction = interaction.get("user_satisfaction", None)
            if user_satisfaction is not None:
                if user_satisfaction > 0.7:
                    mode_success[mode]["success"] += 1
                elif user_satisfaction < 0.3:
                    mode_success[mode]["failure"] += 1
        
        for feedback in feedbacks:
            mode_feedback[feedback.mode] += feedback.engagement_level
            if feedback.engagement_level > 0.6:
                mode_success[feedback.mode]["success"] += 1
            elif feedback.engagement_level < 0.3:
                mode_success[feedback.mode]["failure"] += 1
        
        hot_topics = [k for k, v in topic_freq.items() if v >= 2]
        for topic in hot_topics[:5]:
            insights.append(f"用户频繁提到: {topic}（{topic_freq[topic]}次）")
            self.memory.deep.add_life_theme(topic, importance=min(0.8, topic_freq[topic] * 0.1))
        
        if emotion_trends:
            dominant_emotion = max(set(emotion_trends), key=emotion_trends.count)
            emotion_ratio = emotion_trends.count(dominant_emotion) / len(emotion_trends)
            
            if emotion_ratio > 0.5:
                if dominant_emotion in [EmotionLabels.HAPPY, EmotionLabels.EXCITED]:
                    personality_adjustments.append({"trait": "playfulness", "delta": 0.03})
                    personality_adjustments.append({"trait": "warmth", "delta": 0.02})
                elif dominant_emotion in [EmotionLabels.SAD, EmotionLabels.ANXIOUS]:
                    personality_adjustments.append({"trait": "warmth", "delta": 0.03})
                    personality_adjustments.append({"trait": "philosophical", "delta": 0.02})
                insights.append(f"近期情绪主导: {dominant_emotion}（{int(emotion_ratio*100)}%）")
        
        for mode, stats in mode_success.items():
            total = stats["success"] + stats["failure"]
            if total >= 3:
                success_rate = stats["success"] / total
                mode_feedback[mode] = success_rate
                
                if success_rate > 0.7:
                    insights.append(f"{mode}模式表现优秀（成功率{int(success_rate*100)}%）")
                    new_habits.append({
                        "trigger": f"类似情境",
                        "behavior": f"继续使用{mode}模式",
                        "effectiveness": success_rate
                    })
                elif success_rate < 0.3:
                    insights.append(f"{mode}模式可能需要调整（成功率{int(success_rate*100)}%）")
                    personality_adjustments.append({
                        "trait": self._mode_to_trait(mode),
                        "delta": -0.02
                    })
        
        for hour, modes in time_patterns.items():
            if len(modes) >= 3:
                dominant_mode = max(set(modes), key=modes.count)
                insights.append(f"在{hour}点时，{dominant_mode}模式更有效")
        
        avg_response_time = self._calculate_avg_response_time(feedbacks)
        if avg_response_time > 0:
            if avg_response_time > 5.0:
                insights.append(f"响应时间偏长（{avg_response_time:.1f}秒），建议加快")
                personality_adjustments.append({"trait": "conciseness", "delta": 0.03})
        
        return ReflectionResult(
            memory_updates=memory_updates,
            personality_adjustments=personality_adjustments,
            new_habits=new_habits,
            insights=insights,
            mode_feedback=dict(mode_feedback)
        )
    
    def _mode_to_trait(self, mode: str) -> str:
        mapping = {
            StateModes.PROFESSIONAL: "warmth",
            StateModes.PHILOSOPHICAL: "philosophical",
            StateModes.SLACKING: "slacking_tendency",
            StateModes.MEME: "playfulness",
            StateModes.CONCISE: "conciseness",
        }
        return mapping.get(mode, "warmth")
    
    def _calculate_avg_response_time(self, feedbacks: List[BehaviorFeedback]) -> float:
        if not feedbacks:
            return 0.0
        total = sum(f.response_time for f in feedbacks)
        return total / len(feedbacks)
    
    async def _apply_reflection(self, result: ReflectionResult) -> None:
        for update in result.memory_updates:
            if update["type"] == "important":
                self.memory.long_term.add(
                    content=update["content"],
                    importance=update["importance"],
                    emotion_tag=update["emotion"],
                    source="reflection"
                )
        
        for adjustment in result.personality_adjustments:
            self.personality.update_trait(
                adjustment["trait"],
                adjustment["delta"]
            )
        
        for habit in result.new_habits:
            self.personality.add_habit(
                habit["trigger"],
                habit["behavior"],
                habit.get("effectiveness", 0.5)
            )
        
        for mode, score in result.mode_feedback.items():
            self.personality.record_mode_usage(mode, score if score > 0 else None)
        
        for insight in result.insights:
            print(f"? 反思洞察: {insight}")
    
    async def _save_reflection_log(self, interactions: List[Dict[str, Any]],
                                    result: ReflectionResult) -> None:
        log_file = self.log_path / f"reflection_{int(time.time())}.json"
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "interaction_count": len(interactions),
            "interactions": interactions[-20:],
            "mode_performance": dict(self.mode_performance),
            "result": {
                "memory_updates": result.memory_updates,
                "personality_adjustments": result.personality_adjustments,
                "new_habits": result.new_habits,
                "insights": result.insights,
                "mode_feedback": result.mode_feedback
            }
        }
        
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)


class WeightAdjustmentEngine:
    def __init__(self, decision_engine: StateDecisionEngine):
        self.decision_engine = decision_engine
        self.adjustment_history = []
        self.learning_rate = 0.05
        self.momentum = 0.9
    
    def adjust_intent_weights(self, intent: str, mode: str, reward: float) -> None:
        if intent not in self.decision_engine.intent_weights:
            return
        
        current_weight = self.decision_engine.intent_weights[intent].get(mode, 1.0)
        
        if reward > 0:
            new_weight = current_weight * (1 + self.learning_rate * reward)
        else:
            new_weight = current_weight * (1 - self.learning_rate * abs(reward))
        
        new_weight = max(0.1, min(5.0, new_weight))
        self.decision_engine.intent_weights[intent][mode] = new_weight
        
        self.adjustment_history.append({
            "timestamp": time.time(),
            "type": "intent",
            "intent": intent,
            "mode": mode,
            "old_weight": current_weight,
            "new_weight": new_weight,
            "reward": reward
        })
    
    def adjust_emotion_weights(self, emotion: str, mode: str, reward: float) -> None:
        if emotion not in self.decision_engine.emotion_weights:
            return
        
        current_weight = self.decision_engine.emotion_weights[emotion].get(mode, 1.0)
        
        if reward > 0:
            new_weight = current_weight * (1 + self.learning_rate * reward)
        else:
            new_weight = current_weight * (1 - self.learning_rate * abs(reward))
        
        new_weight = max(0.1, min(3.0, new_weight))
        self.decision_engine.emotion_weights[emotion][mode] = new_weight
        
        self.adjustment_history.append({
            "timestamp": time.time(),
            "type": "emotion",
            "emotion": emotion,
            "mode": mode,
            "old_weight": current_weight,
            "new_weight": new_weight,
            "reward": reward
        })
    
    def get_weight_statistics(self) -> Dict[str, Any]:
        return {
            "intent_weights": {k: dict(v) for k, v in self.decision_engine.intent_weights.items()},
            "emotion_weights": {k: dict(v) for k, v in self.decision_engine.emotion_weights.items()},
            "adjustment_count": len(self.adjustment_history),
            "recent_adjustments": self.adjustment_history[-10:]
        }


class DailySummaryEngine:
    def __init__(self, memory: MemorySystem):
        self.memory = memory
        self.summary_path = Path("./data/daily_summaries")
        self.summary_path.mkdir(parents=True, exist_ok=True)
    
    async def generate_daily_summary(self) -> Dict[str, Any]:
        today = datetime.now().strftime("%Y-%m-%d")
        
        all_memories = self.memory.long_term.get_all()
        
        today_start = time.mktime(datetime.now().replace(hour=0, minute=0, second=0).timetuple())
        today_memories = [
            m for m in all_memories
            if m.get("timestamp", 0) >= today_start
        ]
        
        emotion_counts = defaultdict(int)
        for m in today_memories:
            emotion = m.get("emotion_tag", "neutral")
            emotion_counts[emotion] += 1
        
        important_memories = [
            m for m in today_memories
            if m.get("importance", 0) >= 0.7
        ]
        
        hour_distribution = defaultdict(int)
        for m in today_memories:
            hour = datetime.fromtimestamp(m.get("timestamp", time.time())).hour
            hour_distribution[hour] += 1
        
        peak_hour = max(hour_distribution.items(), key=lambda x: x[1])[0] if hour_distribution else None
        
        summary = {
            "date": today,
            "total_interactions": len(today_memories),
            "emotion_distribution": dict(emotion_counts),
            "important_moments": [m["content"] for m in important_memories[:5]],
            "themes": await self._extract_themes(today_memories),
            "peak_activity_hour": peak_hour,
            "hour_distribution": dict(hour_distribution)
        }
        
        summary_file = self.summary_path / f"summary_{today}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        return summary
    
    async def _extract_themes(self, memories: List[Dict[str, Any]]) -> List[str]:
        keywords = defaultdict(int)
        stopwords = {"的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个"}
        
        for m in memories:
            content = m.get("content", "")
            words = content.replace("，", " ").replace("。", " ").replace("？", " ").split()
            for word in words:
                if len(word) >= 2 and word not in stopwords:
                    keywords[word] += 1
        
        sorted_themes = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        return [theme[0] for theme in sorted_themes[:10]]
    
    def get_weekly_trend(self) -> Dict[str, Any]:
        summaries = []
        for i in range(7):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            summary_file = self.summary_path / f"summary_{date}.json"
            if summary_file.exists():
                with open(summary_file, "r", encoding="utf-8") as f:
                    summaries.append(json.load(f))
        
        if not summaries:
            return {}
        
        total_interactions = sum(s.get("total_interactions", 0) for s in summaries)
        avg_interactions = total_interactions / len(summaries)
        
        emotion_trends = defaultdict(list)
        for s in summaries:
            for emotion, count in s.get("emotion_distribution", {}).items():
                emotion_trends[emotion].append(count)
        
        return {
            "period": "7天",
            "total_interactions": total_interactions,
            "avg_daily_interactions": avg_interactions,
            "emotion_trends": dict(emotion_trends),
            "summary_count": len(summaries)
        }


class PersonalityEvolutionEngine:
    def __init__(self, personality: PersonalityCore):
        self.personality = personality
        self.evolution_log_path = Path("./data/evolution_logs")
        self.evolution_log_path.mkdir(parents=True, exist_ok=True)
        
        self.evolution_history = []
        self.target_traits = {}
    
    def set_target_traits(self, targets: Dict[str, float]) -> None:
        self.target_traits = targets
    
    def evolve_personality(self, feedback: Dict[str, Any]) -> Dict[str, float]:
        adjustments = feedback.get("adjustments", {})
        applied_adjustments = {}
        
        for trait, delta in adjustments.items():
            if abs(delta) > 0.005:
                current = self.personality.traits.get(trait, 5.0)
                target = self.target_traits.get(trait)
                
                if target is not None:
                    direction = target - current
                    effective_delta = delta * (1 if direction * delta > 0 else 0.5)
                else:
                    effective_delta = delta * 0.1
                
                self.personality.update_trait(trait, effective_delta)
                applied_adjustments[trait] = effective_delta
        
        self._log_evolution(applied_adjustments)
        return applied_adjustments
    
    def adaptive_evolution(self, mode_performance: Dict[str, Dict[str, int]]) -> Dict[str, float]:
        adjustments = {}
        
        mode_trait_map = {
            StateModes.PROFESSIONAL: ("warmth", 0.5),
            StateModes.PHILOSOPHICAL: ("philosophical", 0.5),
            StateModes.SLACKING: ("slacking_tendency", 0.5),
            StateModes.MEME: ("playfulness", 0.5),
            StateModes.CONCISE: ("conciseness", 0.5),
        }
        
        for mode, perf in mode_performance.items():
            total = perf.get("success", 0) + perf.get("failure", 0)
            if total < 3:
                continue
            
            success_rate = perf["success"] / total
            trait, base_value = mode_trait_map.get(mode, ("warmth", 0.5))
            
            if success_rate > 0.7:
                adjustments[trait] = adjustments.get(trait, 0) + 0.02
            elif success_rate < 0.3:
                adjustments[trait] = adjustments.get(trait, 0) - 0.02
        
        if adjustments:
            return self.evolve_personality({"adjustments": adjustments})
        return {}
    
    def _log_evolution(self, adjustments: Dict[str, float]) -> None:
        log_file = self.evolution_log_path / f"evolution_{int(time.time())}.json"
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "current_traits": self.personality.get_traits(),
            "adjustments_made": adjustments,
            "target_traits": self.target_traits
        }
        
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        self.evolution_history.append(log_data)
        if len(self.evolution_history) > 100:
            self.evolution_history = self.evolution_history[-100:]
    
    def get_evolution_trend(self, trait: str, days: int = 7) -> List[Dict[str, Any]]:
        cutoff = time.time() - days * 24 * 3600
        relevant = [h for h in self.evolution_history if h.get("timestamp", 0) >= cutoff]
        
        trend = []
        for entry in relevant:
            traits = entry.get("current_traits", {})
            if trait in traits:
                trend.append({
                    "time": entry.get("timestamp"),
                    "value": traits[trait]
                })
        
        return trend
