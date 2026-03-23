import os
import json
import random
import torch
from torch.utils.data import Dataset, IterableDataset, ConcatDataset as TorchConcatDataset
from typing import Optional, List, Dict, Union, Iterator
from pathlib import Path


class TextDataset(Dataset):
    """
    文本数据集
    支持JSONL、TXT等格式
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer,
        max_seq_length: int = 4096,
        shuffle: bool = True,
        seed: int = 42
    ):
        super().__init__()
        
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.shuffle = shuffle
        self.seed = seed
        
        self.examples = self._load_data()
        
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self.examples)
    
    def _load_data(self) -> List[Dict]:
        """加载数据"""
        examples = []
        
        if self.data_path.is_dir():
            files = list(self.data_path.glob("**/*.jsonl")) + list(self.data_path.glob("**/*.txt"))
        else:
            files = [self.data_path]
        
        for file_path in files:
            if file_path.suffix == ".jsonl":
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                if "text" in data:
                                    examples.append({"text": data["text"]})
                                elif "content" in data:
                                    examples.append({"text": data["content"]})
                            except json.JSONDecodeError:
                                continue
            elif file_path.suffix == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    paragraphs = text.split("\n\n")
                    for para in paragraphs:
                        para = para.strip()
                        if para and len(para) > 10:
                            examples.append({"text": para})
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        text = example["text"]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }


class StreamingTextDataset(IterableDataset):
    """
    流式文本数据集
    适用于超大规模数据集
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer,
        max_seq_length: int = 4096,
        buffer_size: int = 10000,
        shuffle: bool = True
    ):
        super().__init__()
        
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        
        self.files = self._get_files()
    
    def _get_files(self) -> List[Path]:
        """获取所有数据文件"""
        if self.data_path.is_dir():
            files = list(self.data_path.glob("**/*.jsonl")) + list(self.data_path.glob("**/*.txt"))
        else:
            files = [self.data_path]
        return files
    
    def _process_line(self, line: str) -> Optional[Dict]:
        """处理单行数据"""
        line = line.strip()
        if not line:
            return None
        
        try:
            data = json.loads(line)
            text = data.get("text") or data.get("content", "")
        except json.JSONDecodeError:
            text = line
        
        if not text or len(text) < 10:
            return None
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }
    
    def __iter__(self) -> Iterator[Dict]:
        """迭代数据"""
        buffer = []
        
        for file_path in self.files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    example = self._process_line(line)
                    if example is not None:
                        buffer.append(example)
                        
                        if len(buffer) >= self.buffer_size:
                            if self.shuffle:
                                random.shuffle(buffer)
                            yield from buffer
                            buffer = []
        
        if buffer:
            if self.shuffle:
                random.shuffle(buffer)
            yield from buffer


class ConversationDataset(Dataset):
    """
    对话数据集
    适用于日常对话训练数据
    格式：[{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}]
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer,
        max_seq_length: int = 4096,
        conversation_template: Optional[str] = None
    ):
        super().__init__()
        
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        self.conversation_template = conversation_template or self._default_template
        
        self.examples = self._load_data()
    
    def _default_template(self, conversations: List[Dict]) -> str:
        """默认对话模板"""
        formatted = []
        for turn in conversations:
            role = turn.get("role", "")
            content = turn.get("content", "")
            
            if role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
            elif role == "system":
                formatted.append(f"System: {content}")
        
        return "\n".join(formatted)
    
    def _load_data(self) -> List[Dict]:
        """加载对话数据"""
        examples = []
        
        if self.data_path.is_dir():
            files = list(self.data_path.glob("**/*.jsonl")) + list(self.data_path.glob("**/*.json"))
        else:
            files = [self.data_path]
        
        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.suffix == ".jsonl":
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                if "conversations" in data:
                                    examples.append(data)
                            except json.JSONDecodeError:
                                continue
                else:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            examples.extend(data)
                        elif "conversations" in data:
                            examples.append(data)
                    except json.JSONDecodeError:
                        continue
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        conversations = example.get("conversations", [])
        
        text = self.conversation_template(conversations)
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class ConcatDataset(TorchConcatDataset):
    """拼接多个数据集"""
    
    def __init__(self, datasets: List[Dataset]):
        super().__init__(datasets)


class DataCollatorForLM:
    """数据整理器"""
    
    def __init__(self, tokenizer, mlm: bool = False):
        self.tokenizer = tokenizer
        self.mlm = mlm
    
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = {
            "input_ids": torch.stack([e["input_ids"] for e in examples]),
            "attention_mask": torch.stack([e["attention_mask"] for e in examples]),
            "labels": torch.stack([e["labels"] for e in examples])
        }
        
        return batch


def create_dataset(
    data_path: str,
    tokenizer,
    max_seq_length: int = 4096,
    dataset_type: str = "text",
    **kwargs
) -> Dataset:
    """
    创建数据集
    
    Args:
        data_path: 数据路径
        tokenizer: 分词器
        max_seq_length: 最大序列长度
        dataset_type: 数据集类型 ("text", "conversation", "streaming")
        **kwargs: 其他参数
    
    Returns:
        Dataset实例
    """
    if dataset_type == "text":
        return TextDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            **kwargs
        )
    elif dataset_type == "conversation":
        return ConversationDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            **kwargs
        )
    elif dataset_type == "streaming":
        return StreamingTextDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
