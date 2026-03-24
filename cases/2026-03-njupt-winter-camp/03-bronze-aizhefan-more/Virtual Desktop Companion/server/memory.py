import json
import time
import hashlib
import random
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path

import redis
import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings


@dataclass
class MemoryItem:
    id: str
    content: str
    timestamp: float
    importance: float
    emotion_tag: str
    tags: List[str]
    fuzzy_time: str = ""
    confidence: float = 1.0
    source: str = "user"
    decay_factor: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        return cls(**data)


class WorkingMemory:
    def __init__(self, max_size: int = None):
        self.max_size = max_size or settings.WORKING_MEMORY_SIZE
        self._storage = {}
        self._use_redis = False
        
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True
            )
            self.redis_client.ping()
            self._use_redis = True
            print("WorkingMemory using Redis")
        except Exception as e:
            print(f"WorkingMemory using memory (Redis unavailable: {e})")
    
    def _get_key(self, session_id: str) -> str:
        return f"working_memory:{session_id}"
    
    def add(self, session_id: str, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        key = self._get_key(session_id)
        message["timestamp"] = time.time()
        
        if self._use_redis:
            current = self.redis_client.get(key)
            messages = json.loads(current) if current else []
            messages.append(message)
            if len(messages) > self.max_size:
                messages = messages[-self.max_size:]
            self.redis_client.setex(key, 3600, json.dumps(messages, ensure_ascii=False))
        else:
            if key not in self._storage:
                self._storage[key] = []
            self._storage[key].append(message)
            if len(self._storage[key]) > self.max_size:
                self._storage[key] = self._storage[key][-self.max_size:]
            messages = self._storage[key]
        
        return messages
    
    def get(self, session_id: str) -> List[Dict[str, Any]]:
        key = self._get_key(session_id)
        
        if self._use_redis:
            current = self.redis_client.get(key)
            return json.loads(current) if current else []
        else:
            return self._storage.get(key, [])
    
    def clear(self, session_id: str) -> None:
        key = self._get_key(session_id)
        
        if self._use_redis:
            self.redis_client.delete(key)
        else:
            if key in self._storage:
                del self._storage[key]
    
    def format_for_prompt(self, session_id: str) -> str:
        messages = self.get(session_id)
        if not messages:
            return ""
        
        lines = []
        for msg in messages[-7:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                lines.append(f"用户: {content}")
            else:
                lines.append(f"AI: {content}")
        
        return "\n".join(lines)


class LongTermMemory:
    def __init__(self):
        self.db_path = Path(settings.VECTOR_DB_PATH)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name="long_term_memory",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.decay_rate = settings.MEMORY_DECAY_RATE
        self.fuzzy_factor = settings.MEMORY_FUZZY_FACTOR
    
    def _generate_id(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()
    
    def _fuzzy_time(self, timestamp: float) -> str:
        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour
        
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
        
        weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][dt.weekday()]
        season = "春天" if 3 <= dt.month <= 5 else "夏天" if 6 <= dt.month <= 8 else "秋天" if 9 <= dt.month <= 11 else "冬天"
        
        return f"{season}某个{weekday}的{time_of_day}"
    
    def _fuzzy_content(self, content: str) -> str:
        if random.random() > self.fuzzy_factor:
            return content
        
        words = content.split()
        if len(words) > 5:
            indices = random.sample(range(len(words)), min(2, len(words) // 5))
            for i in indices:
                words[i] = "..."
            return " ".join(words)
        return content
    
    def _calculate_decay(self, timestamp: float, importance: float) -> float:
        age_seconds = time.time() - timestamp
        age_days = age_seconds / (24 * 3600)
        
        decay = math.exp(-self.decay_rate * age_days * (1 - importance))
        return decay
    
    def add(self, content: str, importance: float = 0.5, 
            emotion_tag: str = "neutral", tags: List[str] = None,
            source: str = "user") -> str:
        
        memory_id = self._generate_id(content)
        timestamp = time.time()
        fuzzy_time = self._fuzzy_time(timestamp)
        
        metadata = {
            "timestamp": timestamp,
            "importance": importance,
            "emotion_tag": emotion_tag,
            "fuzzy_time": fuzzy_time,
            "source": source,
            "decay_factor": 1.0,
            "access_count": 0,
            "last_access": timestamp
        }
        if tags:
            for i, tag in enumerate(tags[:5]):
                metadata[f"tag_{i}"] = tag
        
        document = f"{fuzzy_time}，{content}"
        
        self.collection.add(
            ids=[memory_id],
            documents=[document],
            metadatas=[metadata]
        )
        
        return memory_id
    
    def _update_access(self, memory_id: str, metadata: Dict[str, Any]) -> None:
        new_metadata = metadata.copy()
        new_metadata["access_count"] = metadata.get("access_count", 0) + 1
        new_metadata["last_access"] = time.time()
        
        importance = metadata.get("importance", 0.5)
        access_count = new_metadata["access_count"]
        new_importance = min(1.0, importance + 0.01 * access_count)
        new_metadata["importance"] = new_importance
        
        self.collection.update(
            ids=[memory_id],
            metadatas=[new_metadata]
        )
    
    def search(self, query: str, top_k: int = None, 
               emotion_filter: str = None,
               min_importance: float = 0.0) -> List[Dict[str, Any]]:
        
        top_k = top_k or settings.LONG_TERM_MEMORY_TOP_K
        
        where = {}
        if emotion_filter:
            where["emotion_tag"] = emotion_filter
        if min_importance > 0:
            where["importance"] = {"$gte": min_importance}
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k * 2,
            where=where if where else None
        )
        
        memories = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i] if results["distances"] else 0
                
                timestamp = metadata.get("timestamp", time.time())
                importance = metadata.get("importance", 0.5)
                decay = self._calculate_decay(timestamp, importance)
                
                base_confidence = 1.0 - distance
                adjusted_confidence = base_confidence * decay
                
                if adjusted_confidence > 0.1:
                    fuzzy_content = self._fuzzy_content(doc)
                    
                    memories.append({
                        "id": results["ids"][0][i],
                        "content": fuzzy_content,
                        "original_content": doc,
                        "timestamp": timestamp,
                        "importance": importance,
                        "emotion_tag": metadata.get("emotion_tag", "neutral"),
                        "fuzzy_time": metadata.get("fuzzy_time", ""),
                        "confidence": adjusted_confidence,
                        "decay_factor": decay,
                        "source": metadata.get("source", "user"),
                        "access_count": metadata.get("access_count", 0)
                    })
        
        memories = sorted(memories, key=lambda x: x["confidence"], reverse=True)
        return memories[:top_k]
    
    def format_for_prompt(self, query: str, top_k: int = 3) -> str:
        memories = self.search(query, top_k=top_k)
        if not memories:
            return ""
        
        lines = ["我记得一些相关的事情："]
        for mem in memories:
            conf = mem["confidence"]
            if conf > 0.8:
                prefix = "我清楚地记得"
            elif conf > 0.6:
                prefix = "我好像记得"
            elif conf > 0.4:
                prefix = "我隐约记得"
            else:
                prefix = "我好像有点印象，"
            
            fuzzy_time = mem.get("fuzzy_time", "")
            if fuzzy_time and random.random() > 0.3:
                time_hint = f"{fuzzy_time}，"
            else:
                time_hint = ""
            
            lines.append(f"- {prefix}{time_hint}{mem['content']}")
        
        return "\n".join(lines)
    
    def get_all(self) -> List[Dict[str, Any]]:
        results = self.collection.get()
        memories = []
        for i, doc in enumerate(results["documents"]):
            metadata = results["metadatas"][i]
            memories.append({
                "id": results["ids"][i],
                "content": doc,
                "timestamp": metadata.get("timestamp"),
                "importance": metadata.get("importance", 0.5),
                "emotion_tag": metadata.get("emotion_tag", "neutral"),
                "access_count": metadata.get("access_count", 0)
            })
        return memories
    
    def delete(self, memory_id: str) -> None:
        self.collection.delete(ids=[memory_id])
    
    def cleanup_old_memories(self, max_age_days: int = 90) -> int:
        all_memories = self.get_all()
        cutoff_time = time.time() - max_age_days * 24 * 3600
        
        deleted = 0
        for mem in all_memories:
            timestamp = mem.get("timestamp", time.time())
            importance = mem.get("importance", 0.5)
            
            if timestamp < cutoff_time and importance < 0.6:
                self.delete(mem["id"])
                deleted += 1
        
        return deleted


class DeepMemory:
    def __init__(self):
        self.db_path = Path(settings.MEMORY_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.data_file = self.db_path
        
        if self.data_file.exists():
            with open(self.data_file, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = {
                "user_profile": {},
                "life_themes": [],
                "relationship_graph": {},
                "core_memories": [],
                "interaction_patterns": [],
                "preferences": {}
            }
    
    def _save(self) -> None:
        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def update_user_profile(self, key: str, value: Any, confidence: float = 0.8) -> None:
        if key not in self.data["user_profile"]:
            self.data["user_profile"][key] = {
                "value": value,
                "confidence": confidence,
                "count": 1,
                "last_updated": time.time()
            }
        else:
            existing = self.data["user_profile"][key]
            new_confidence = (existing["confidence"] * existing["count"] + confidence) / (existing["count"] + 1)
            self.data["user_profile"][key] = {
                "value": value,
                "confidence": new_confidence,
                "count": existing["count"] + 1,
                "last_updated": time.time()
            }
        self._save()
    
    def get_user_profile(self) -> Dict[str, Any]:
        profile = self.data.get("user_profile", {})
        return {k: v["value"] for k, v in profile.items() if v.get("confidence", 0) > 0.5}
    
    def add_life_theme(self, theme: str, importance: float = 0.5) -> None:
        themes = self.data.get("life_themes", [])
        existing = next((t for t in themes if t["theme"] == theme), None)
        
        if existing:
            existing["importance"] = max(existing["importance"], importance)
            existing["count"] = existing.get("count", 0) + 1
            existing["last_mentioned"] = time.time()
        else:
            themes.append({
                "theme": theme,
                "importance": importance,
                "count": 1,
                "first_mentioned": time.time(),
                "last_mentioned": time.time()
            })
        
        self.data["life_themes"] = sorted(themes, key=lambda x: (x["importance"], x["count"]), reverse=True)[:20]
        self._save()
    
    def get_life_themes(self, top_n: int = 5) -> List[Dict[str, Any]]:
        themes = self.data.get("life_themes", [])
        return themes[:top_n]
    
    def add_core_memory(self, content: str, emotion: str = "neutral", 
                        importance: float = 0.8) -> None:
        core_memories = self.data.get("core_memories", [])
        
        existing = next((m for m in core_memories if m["content"] == content), None)
        if existing:
            existing["importance"] = max(existing["importance"], importance)
            existing["recall_count"] = existing.get("recall_count", 0) + 1
        else:
            core_memories.append({
                "content": content,
                "emotion": emotion,
                "importance": importance,
                "timestamp": time.time(),
                "recall_count": 1
            })
        
        self.data["core_memories"] = sorted(core_memories, key=lambda x: (x["importance"], x.get("recall_count", 1)), reverse=True)[:100]
        self._save()
    
    def get_core_memories(self) -> List[Dict[str, Any]]:
        return self.data.get("core_memories", [])
    
    def add_interaction_pattern(self, trigger: str, response_type: str, 
                                 effectiveness: float = 0.5) -> None:
        patterns = self.data.get("interaction_patterns", [])
        
        existing = next((p for p in patterns if p["trigger"] == trigger), None)
        if existing:
            existing["count"] = existing.get("count", 0) + 1
            existing["effectiveness"] = (existing["effectiveness"] * existing["count"] + effectiveness) / (existing["count"] + 1)
        else:
            patterns.append({
                "trigger": trigger,
                "response_type": response_type,
                "effectiveness": effectiveness,
                "count": 1,
                "created_at": time.time()
            })
        
        self.data["interaction_patterns"] = sorted(patterns, key=lambda x: (x["effectiveness"], x["count"]), reverse=True)[:30]
        self._save()
    
    def update_preference(self, category: str, item: str, preference: float) -> None:
        if category not in self.data["preferences"]:
            self.data["preferences"][category] = {}
        
        self.data["preferences"][category][item] = preference
        self._save()
    
    def get_preferences(self, category: str = None) -> Dict[str, Any]:
        if category:
            return self.data.get("preferences", {}).get(category, {})
        return self.data.get("preferences", {})
    
    def format_for_prompt(self) -> str:
        parts = []
        
        profile = self.get_user_profile()
        if profile:
            high_confidence = {k: v for k, v in profile.items() if self.data["user_profile"].get(k, {}).get("confidence", 0) > 0.7}
            if high_confidence:
                profile_str = "、".join([f"{k}:{v}" for k, v in list(high_confidence.items())[:5]])
                parts.append(f"用户画像：{profile_str}")
        
        themes = self.get_life_themes(3)
        if themes:
            theme_str = "、".join([t["theme"] for t in themes])
            parts.append(f"用户常谈论的话题：{theme_str}")
        
        patterns = self.data.get("interaction_patterns", [])[:3]
        if patterns:
            pattern_desc = []
            for p in patterns:
                if p.get("effectiveness", 0) > 0.6:
                    pattern_desc.append(f"当{p['trigger']}时，用户喜欢{p['response_type']}")
            if pattern_desc:
                parts.append("用户习惯：" + "；".join(pattern_desc))
        
        return "\n".join(parts) if parts else ""


class MemorySystem:
    def __init__(self):
        self.working = WorkingMemory()
        self.long_term = LongTermMemory()
        self.deep = DeepMemory()
    
    def add_interaction(self, session_id: str, user_input: str, 
                        ai_response: str, importance: float = 0.5,
                        emotion_tag: str = "neutral", tags: List[str] = None) -> None:
        
        self.working.add(session_id, {"role": "user", "content": user_input})
        self.working.add(session_id, {"role": "assistant", "content": ai_response})
        
        if importance >= 0.3:
            self.long_term.add(
                content=f"用户说：{user_input}，我回答：{ai_response}",
                importance=importance,
                emotion_tag=emotion_tag,
                tags=tags,
                source="interaction"
            )
        
        if importance >= 0.7:
            self.deep.add_core_memory(
                content=user_input,
                emotion=emotion_tag,
                importance=importance
            )
    
    def retrieve_context(self, session_id: str, query: str) -> Dict[str, str]:
        working_context = self.working.format_for_prompt(session_id)
        long_term_context = self.long_term.format_for_prompt(query)
        deep_context = self.deep.format_for_prompt()
        
        return {
            "working": working_context,
            "long_term": long_term_context,
            "deep": deep_context
        }
    
    def extract_themes(self, text: str, keywords: List[str]) -> List[str]:
        themes = []
        
        theme_keywords = {
            "工作": ["工作", "上班", "加班", "项目", "会议", "同事", "老板"],
            "学习": ["学习", "考试", "作业", "论文", "课程", "学校"],
            "生活": ["生活", "日常", "今天", "明天", "周末", "假期"],
            "情感": ["喜欢", "难过", "开心", "爱情", "朋友", "家人"],
            "技术": ["编程", "代码", "技术", "算法", "系统", "软件"],
            "健康": ["健康", "身体", "锻炼", "饮食", "睡眠", "医疗"],
        }
        
        for theme, keywords in theme_keywords.items():
            for kw in keywords:
                if kw in text:
                    themes.append(theme)
                    break
        
        return list(set(themes))
    
    def learn_from_interaction(self, user_input: str, ai_response: str, 
                                user_feedback: float = None) -> None:
        if user_feedback is not None and user_feedback > 0.5:
            themes = self.extract_themes(user_input, [])
            for theme in themes:
                self.deep.add_life_theme(theme, importance=user_feedback)
    
    def clear_session(self, session_id: str) -> None:
        self.working.clear(session_id)
