import json
import time
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings


class RAGSystem:
    def __init__(self):
        self.db_path = Path(settings.VECTOR_DB_PATH) / "rag"
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name="professional_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.initialized = False
        self._cache = {}  # 缓存检索结果
        self._cache_size = 100  # 缓存大小
    
    def initialize(self):
        """初始化RAG系统，加载专业知识"""
        if self.initialized:
            return
        
        # 加载WSL2专业知识
        wsl2_path = Path("data_collection/processed_data/wsl2_context_enhancement.json")
        if wsl2_path.exists():
            self._load_wsl2_knowledge(wsl2_path)
        
        # 加载其他专业知识文件
        all_data_path = Path("data_collection/processed_data/all_data.json")
        if all_data_path.exists():
            self._load_all_data(all_data_path)
        
        self.initialized = True
    
    def _load_wsl2_knowledge(self, file_path: Path):
        """加载WSL2专业知识"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        knowledge_summary = data.get("knowledge_summary", "")
        if not knowledge_summary:
            return
        
        # 解析知识内容，按章节分割
        sections = self._parse_knowledge_sections(knowledge_summary)
        for section_title, section_content in sections.items():
            self._add_knowledge(
                content=section_content,
                category="WSL2",
                subcategory=section_title,
                importance=0.9
            )
    
    def _load_all_data(self, file_path: Path):
        """加载所有专业知识数据"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for item in data:
                content = item.get("content", "")
                category = item.get("category", "general")
                subcategory = item.get("subcategory", "")
                importance = item.get("importance", 0.7)
                
                if content:
                    self._add_knowledge(
                        content=content,
                        category=category,
                        subcategory=subcategory,
                        importance=importance
                    )
    
    def _parse_knowledge_sections(self, knowledge_summary: str) -> Dict[str, str]:
        """解析知识章节"""
        sections = {}
        lines = knowledge_summary.split("\n")
        current_section = ""
        current_content = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("## "):
                if current_section and current_content:
                    sections[current_section] = "\n".join(current_content)
                current_section = line[3:].strip()
                current_content = []
            elif line:
                current_content.append(line)
        
        if current_section and current_content:
            sections[current_section] = "\n".join(current_content)
        
        return sections
    
    def _generate_id(self, content: str, category: str) -> str:
        """生成知识ID"""
        return hashlib.md5(f"{category}:{content[:100]}".encode()).hexdigest()
    
    def _add_knowledge(self, content: str, category: str, 
                      subcategory: str = "", importance: float = 0.7):
        """添加专业知识到向量库"""
        knowledge_id = self._generate_id(content, category)
        timestamp = time.time()
        
        metadata = {
            "timestamp": timestamp,
            "category": category,
            "subcategory": subcategory,
            "importance": importance,
            "access_count": 0,
            "last_access": timestamp
        }
        
        document = f"【{category}】{subcategory}\n{content}"
        
        try:
            self.collection.add(
                ids=[knowledge_id],
                documents=[document],
                metadatas=[metadata]
            )
        except Exception as e:
            print(f"添加知识失败: {e}")
    
    def _get_cache_key(self, query: str, category: Optional[str] = None, 
                      top_k: int = 3, min_importance: float = 0.5) -> str:
        """生成缓存键"""
        return f"{query}:{category}:{top_k}:{min_importance}"
    
    def search(self, query: str, category: Optional[str] = None, 
               top_k: int = 3, min_importance: float = 0.5) -> List[Dict[str, Any]]:
        """搜索专业知识"""
        # 检查缓存
        cache_key = self._get_cache_key(query, category, top_k, min_importance)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if not self.initialized:
            self.initialize()
        
        where = {}
        if category:
            where["category"] = category
        if min_importance > 0:
            where["importance"] = {"$gte": min_importance}
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where if where else None
            )
            
            knowledge_items = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i] if results["distances"] else 0
                    
                    confidence = max(0, 1.0 - distance)
                    if confidence > 0.3:
                        knowledge_items.append({
                            "id": results["ids"][0][i],
                            "content": doc,
                            "category": metadata.get("category", "general"),
                            "subcategory": metadata.get("subcategory", ""),
                            "importance": metadata.get("importance", 0.7),
                            "confidence": confidence,
                            "distance": distance
                        })
            
            # 更新缓存
            self._update_cache(cache_key, knowledge_items)
            return knowledge_items
        except Exception as e:
            print(f"RAG搜索失败: {e}")
            return []
    
    def _update_cache(self, key: str, value: Any):
        """更新缓存"""
        if len(self._cache) >= self._cache_size:
            # 移除最旧的缓存项
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value
    
    def format_for_prompt(self, query: str, category: Optional[str] = None, 
                         top_k: int = 3) -> str:
        """格式化专业知识为prompt"""
        knowledge_items = self.search(query, category, top_k)
        if not knowledge_items:
            return ""
        
        lines = ["【专业知识】"]
        for item in knowledge_items:
            category_str = f"【{item['category']}】"
            if item['subcategory']:
                category_str += f"{item['subcategory']}"
            
            content = item['content'].replace("【" + item['category'] + "】" + item['subcategory'], "").strip()
            
            confidence_str = ""
            if item['confidence'] > 0.8:
                confidence_str = "（确信）"
            elif item['confidence'] > 0.6:
                confidence_str = "（较确信）"
            
            lines.append(f"- {category_str}{confidence_str}")
            lines.append(f"  {content[:500]}{'...' if len(content) > 500 else ''}")
        
        return "\n".join(lines)
    
    def get_categories(self) -> List[str]:
        """获取所有知识分类"""
        if not self.initialized:
            self.initialize()
        
        results = self.collection.get()
        categories = set()
        for metadata in results.get("metadatas", []):
            if metadata and "category" in metadata:
                categories.add(metadata["category"])
        
        return list(categories)
    
    def clear(self):
        """清空所有知识"""
        self.collection.delete(ids=self.collection.get()["ids"])
        self.initialized = False


# 全局RAG系统实例
rag_system = RAGSystem()
