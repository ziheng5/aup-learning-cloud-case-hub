#!/usr/bin/env python3
"""
Wikipedia数据处理脚本
将wikiextractor输出转换为训练格式
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import random
import re

def clean_wikipedia_text(text: str) -> str:
    """清洗Wikipedia文本"""
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'（.*?）', '', text)
    text = re.sub(r'第[零一二三四五六七八九十百0-9]+章', '', text)
    text = re.sub(r'参见.*', '', text)
    text = re.sub(r'参考资料.*', '', text)
    text = re.sub(r'外部链接.*', '', text)
    text = re.sub(r'注释.*', '', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def process_wiki_directory(
    input_dir: str,
    output_dir: str,
    min_length: int = 100,
    max_length: int = 10000,
    eval_ratio: float = 0.001
):
    """
    处理Wikipedia提取后的目录
    
    Args:
        input_dir: wikiextractor输出目录 (extracted/)
        output_dir: 输出目录
        min_length: 最小文本长度
        max_length: 最大文本长度
        eval_ratio: 验证集比例
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "train").mkdir(exist_ok=True)
    (output_path / "eval").mkdir(exist_ok=True)
    
    all_files = list(input_path.rglob("*/wiki_*"))
    
    if not all_files:
        print(f"错误：在 {input_dir} 中未找到wiki文件")
        print("请确认路径正确，文件应该在 extracted/AA/wiki_00 等位置")
        return
    
    print(f"找到 {len(all_files)} 个文件")
    print(f"开始处理...")
    
    all_docs = []
    total_chars = 0
    
    for file_path in tqdm(all_files, desc="处理文件"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        doc = json.loads(line)
                        text = doc.get('text', '')
                        
                        if not text:
                            continue
                        
                        text = clean_wikipedia_text(text)
                        
                        if len(text) < min_length:
                            continue
                        
                        if len(text) > max_length:
                            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
                            for chunk in chunks:
                                if len(chunk) >= min_length:
                                    all_docs.append(chunk)
                                    total_chars += len(chunk)
                        else:
                            all_docs.append(text)
                            total_chars += len(text)
                            
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue
    
    print(f"\n处理完成！")
    print(f"  文档数量: {len(all_docs):,}")
    print(f"  总字符数: {total_chars:,}")
    print(f"  估算大小: {total_chars / 1024 / 1024 / 1024:.2f} GB")
    
    print("\n打乱数据...")
    random.shuffle(all_docs)
    
    eval_size = max(1000, int(len(all_docs) * eval_ratio))
    train_docs = all_docs[eval_size:]
    eval_docs = all_docs[:eval_size]
    
    print(f"\n分割数据集：")
    print(f"  训练集: {len(train_docs):,} 条")
    print(f"  验证集: {len(eval_docs):,} 条")
    
    train_file = output_path / "train" / "wikipedia_zh.jsonl"
    eval_file = output_path / "eval" / "wikipedia_zh.jsonl"
    
    print(f"\n写入训练集: {train_file}")
    with open(train_file, 'w', encoding='utf-8') as f:
        for doc in tqdm(train_docs, desc="写入训练集"):
            f.write(json.dumps({"text": doc}, ensure_ascii=False) + '\n')
    
    print(f"写入验证集: {eval_file}")
    with open(eval_file, 'w', encoding='utf-8') as f:
        for doc in tqdm(eval_docs, desc="写入验证集"):
            f.write(json.dumps({"text": doc}, ensure_ascii=False) + '\n')
    
    train_size = train_file.stat().st_size / 1024 / 1024 / 1024
    eval_size = eval_file.stat().st_size / 1024 / 1024 / 1024
    
    print(f"\n最终输出：")
    print(f"  训练集: {train_file} ({train_size:.2f} GB)")
    print(f"  验证集: {eval_file} ({eval_size:.2f} GB)")
    print(f"  总计: {train_size + eval_size:.2f} GB")
    
    print("\n" + "="*60)
    print("数据准备完成！现在可以开始训练：")
    print(f"python scripts/train.py --train_data {output_path}/train --eval_data {output_path}/eval")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="处理Wikipedia中文数据")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="wikiextractor输出目录，如 extracted/"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="输出目录"
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=100,
        help="最小文本长度（字符）"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=10000,
        help="最大文本长度（字符）"
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.001,
        help="验证集比例"
    )
    
    args = parser.parse_args()
    
    process_wiki_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_length=args.min_length,
        max_length=args.max_length,
        eval_ratio=args.eval_ratio
    )

if __name__ == "__main__":
    main()
