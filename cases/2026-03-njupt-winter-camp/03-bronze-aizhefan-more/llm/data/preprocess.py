#!/usr/bin/env python3
"""
数据预处理工具
支持多种数据源的清洗、过滤和格式化
"""

import os
import json
import re
import random
from pathlib import Path
from typing import List, Dict, Optional, Union
from tqdm import tqdm
import argparse


class DataCleaner:
    """数据清洗器"""
    
    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 100000,
        filter_special_chars: bool = True,
        filter_duplicates: bool = True
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.filter_special_chars = filter_special_chars
        self.filter_duplicates = filter_duplicates
        
        self.seen_texts = set()
    
    def clean(self, text: str) -> Optional[str]:
        """清洗文本"""
        if not text:
            return None
        
        text = text.strip()
        
        if len(text) < self.min_length:
            return None
        
        if len(text) > self.max_length:
            return None
        
        if self.filter_special_chars:
            text = self._remove_special_chars(text)
        
        if self.filter_duplicates:
            text_hash = hash(text)
            if text_hash in self.seen_texts:
                return None
            self.seen_texts.add(text_hash)
        
        text = self._normalize_whitespace(text)
        
        return text if text else None
    
    def _remove_special_chars(self, text: str) -> str:
        """移除特殊字符"""
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        text = re.sub(r'[?]', '', text)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """规范化空白字符"""
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        return text.strip()


class WikipediaProcessor:
    """Wikipedia数据处理器"""
    
    def __init__(self, cleaner: DataCleaner = None):
        self.cleaner = cleaner or DataCleaner()
    
    def process_file(self, input_path: str, output_path: str):
        """处理Wikipedia数据"""
        print(f"Processing Wikipedia data: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line in tqdm(infile, desc="Processing"):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    text = data.get('text', '') or data.get('content', '')
                    
                    cleaned = self.cleaner.clean(text)
                    if cleaned:
                        outfile.write(json.dumps({"text": cleaned}, ensure_ascii=False) + '\n')
                        
                except json.JSONDecodeError:
                    continue


class TextProcessor:
    """纯文本数据处理器"""
    
    def __init__(self, cleaner: DataCleaner = None):
        self.cleaner = cleaner or DataCleaner()
    
    def process_file(self, input_path: str, output_path: str, paragraph_mode: bool = True):
        """处理纯文本文件"""
        print(f"Processing text file: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as infile:
            content = infile.read()
        
        if paragraph_mode:
            paragraphs = re.split(r'\n\s*\n', content)
        else:
            paragraphs = content.split('\n')
        
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for para in tqdm(paragraphs, desc="Processing"):
                cleaned = self.cleaner.clean(para)
                if cleaned:
                    outfile.write(json.dumps({"text": cleaned}, ensure_ascii=False) + '\n')


class ConversationProcessor:
    """对话数据处理器"""
    
    def __init__(self, cleaner: DataCleaner = None):
        self.cleaner = cleaner or DataCleaner()
    
    def process_file(self, input_path: str, output_path: str):
        """处理对话数据"""
        print(f"Processing conversation data: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line in tqdm(infile, desc="Processing"):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    if 'conversations' in data:
                        conversations = data['conversations']
                        formatted = self._format_conversation(conversations)
                        if formatted:
                            outfile.write(json.dumps({"text": formatted}, ensure_ascii=False) + '\n')
                    
                    elif 'messages' in data:
                        messages = data['messages']
                        formatted = self._format_messages(messages)
                        if formatted:
                            outfile.write(json.dumps({"text": formatted}, ensure_ascii=False) + '\n')
                            
                except json.JSONDecodeError:
                    continue
    
    def _format_conversation(self, conversations: List[Dict]) -> str:
        """格式化对话"""
        lines = []
        for turn in conversations:
            role = turn.get('role', '')
            content = turn.get('content', '')
            
            if role == 'user':
                lines.append(f"用户：{content}")
            elif role == 'assistant':
                lines.append(f"助手：{content}")
            elif role == 'system':
                lines.append(f"系统：{content}")
        
        return '\n'.join(lines) if lines else None
    
    def _format_messages(self, messages: List[Dict]) -> str:
        """格式化消息"""
        lines = []
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role == 'user':
                lines.append(f"用户：{content}")
            elif role == 'assistant':
                lines.append(f"助手：{content}")
        
        return '\n'.join(lines) if lines else None


def split_train_eval(
    input_path: str,
    train_path: str,
    eval_path: str,
    eval_ratio: float = 0.01,
    seed: int = 42
):
    """分割训练集和验证集"""
    print(f"Splitting data: {input_path}")
    
    random.seed(seed)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    random.shuffle(lines)
    
    eval_size = int(len(lines) * eval_ratio)
    eval_lines = lines[:eval_size]
    train_lines = lines[eval_size:]
    
    with open(train_path, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    with open(eval_path, 'w', encoding='utf-8') as f:
        f.writelines(eval_lines)
    
    print(f"Train: {len(train_lines)} lines")
    print(f"Eval: {len(eval_lines)} lines")


def merge_files(input_dir: str, output_path: str):
    """合并多个JSONL文件"""
    print(f"Merging files from: {input_dir}")
    
    input_path = Path(input_dir)
    files = list(input_path.glob('*.jsonl'))
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for file_path in tqdm(files, desc="Merging"):
            with open(file_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)


def sample_data(
    input_path: str,
    output_path: str,
    num_samples: int,
    seed: int = 42
):
    """采样数据"""
    print(f"Sampling {num_samples} examples from: {input_path}")
    
    random.seed(seed)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if len(lines) <= num_samples:
        samples = lines
    else:
        samples = random.sample(lines, num_samples)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(samples)
    
    print(f"Sampled {len(samples)} examples")


def main():
    parser = argparse.ArgumentParser(description="数据预处理工具")
    subparsers = parser.add_subparsers(dest='command')
    
    clean_parser = subparsers.add_parser('clean', help='清洗数据')
    clean_parser.add_argument('--input', required=True, help='输入文件路径')
    clean_parser.add_argument('--output', required=True, help='输出文件路径')
    clean_parser.add_argument('--format', choices=['jsonl', 'text', 'conversation'], required=True, help='数据格式')
    clean_parser.add_argument('--min_length', type=int, default=10, help='最小文本长度')
    clean_parser.add_argument('--max_length', type=int, default=100000, help='最大文本长度')
    
    split_parser = subparsers.add_parser('split', help='分割训练集和验证集')
    split_parser.add_argument('--input', required=True, help='输入文件路径')
    split_parser.add_argument('--train', required=True, help='训练集输出路径')
    split_parser.add_argument('--eval', required=True, help='验证集输出路径')
    split_parser.add_argument('--eval_ratio', type=float, default=0.01, help='验证集比例')
    
    merge_parser = subparsers.add_parser('merge', help='合并文件')
    merge_parser.add_argument('--input_dir', required=True, help='输入目录')
    merge_parser.add_argument('--output', required=True, help='输出文件路径')
    
    sample_parser = subparsers.add_parser('sample', help='采样数据')
    sample_parser.add_argument('--input', required=True, help='输入文件路径')
    sample_parser.add_argument('--output', required=True, help='输出文件路径')
    sample_parser.add_argument('--num_samples', type=int, required=True, help='采样数量')
    
    args = parser.parse_args()
    
    if args.command == 'clean':
        cleaner = DataCleaner(
            min_length=args.min_length,
            max_length=args.max_length
        )
        
        if args.format == 'jsonl':
            processor = WikipediaProcessor(cleaner)
        elif args.format == 'text':
            processor = TextProcessor(cleaner)
        else:
            processor = ConversationProcessor(cleaner)
        
        processor.process_file(args.input, args.output)
    
    elif args.command == 'split':
        split_train_eval(args.input, args.train, args.eval, args.eval_ratio)
    
    elif args.command == 'merge':
        merge_files(args.input_dir, args.output)
    
    elif args.command == 'sample':
        sample_data(args.input, args.output, args.num_samples)


if __name__ == '__main__':
    main()
