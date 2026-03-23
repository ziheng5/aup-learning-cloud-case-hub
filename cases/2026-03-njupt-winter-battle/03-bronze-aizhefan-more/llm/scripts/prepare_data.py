#!/usr/bin/env python3
"""
数据准备脚本
用法: python scripts/prepare_data.py --source wiki --output_dir ./data
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess import DataCleaner, WikipediaProcessor, TextProcessor

def create_sample_dataset(output_dir: str, num_examples: int = 1000):
    """创建示例数据集（用于测试）"""
    print(f"Creating sample dataset with {num_examples} examples...")
    
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "eval"), exist_ok=True)
    
    sample_texts = [
        "人工智能是计算机科学的一个分支，它企图了解智能的实质。",
        "机器学习是人工智能的核心，是使计算机具有智能的根本途径。",
        "深度学习是机器学习的一个子领域，它使用多层神经网络。",
        "自然语言处理是人工智能的重要应用领域。",
        "计算机视觉使计算机能够理解和解释视觉信息。",
        "机器人学是研究机器人设计、制造和应用的科学。",
        "专家系统是一种模拟人类专家决策能力的计算机系统。",
        "语音识别使计算机能够识别和理解人类语音。",
        "机器翻译是利用计算机将一种语言翻译成另一种语言。",
        "数据挖掘是从大量数据中提取有用信息的过程。"
    ]
    
    cleaner = DataCleaner(min_length=5)
    
    train_path = os.path.join(output_dir, "train", "sample.jsonl")
    eval_path = os.path.join(output_dir, "eval", "sample.jsonl")
    
    with open(train_path, 'w', encoding='utf-8') as train_f, \
         open(eval_path, 'w', encoding='utf-8') as eval_f:
        
        for i in range(num_examples):
            text = sample_texts[i % len(sample_texts)]
            text = f"{text} 这是第{i+1}条训练数据。" * ((i % 5) + 1)
            
            cleaned = cleaner.clean(text)
            if cleaned:
                if i < num_examples * 0.99:
                    train_f.write(json.dumps({"text": cleaned}, ensure_ascii=False) + '\n')
                else:
                    eval_f.write(json.dumps({"text": cleaned}, ensure_ascii=False) + '\n')
    
    print(f"Sample dataset created:")
    print(f"  Train: {train_path}")
    print(f"  Eval: {eval_path}")

def download_wikipedia(lang: str = "zh", output_dir: str = "./data/raw"):
    """
    下载Wikipedia数据（提示用户手动下载）
    
    注意：Wikipedia数据较大（中文约10GB压缩包），建议手动下载
    """
    print("="*60)
    print("Wikipedia数据下载指南")
    print("="*60)
    print()
    print("由于Wikipedia数据较大，请手动下载：")
    print()
    print("1. 访问：https://dumps.wikimedia.org/zhwiki/")
    print("2. 选择最新的dump日期")
    print("3. 下载以下文件：")
    print("   - zhwiki-YYYYMMDD-pages-articles.xml.bz2")
    print("   - 约10GB压缩包，解压后约50GB+")
    print()
    print("4. 使用wikiextractor提取纯文本：")
    print("   pip install wikiextractor")
    print("   wikiextractor zhwiki-YYYYMMDD-pages-articles.xml.bz2 -b 100M -o extracted")
    print()
    print("5. 转换为JSONL格式：")
    print("   python scripts/prepare_data.py --source wiki --input extracted --output ./data")
    print()

def process_wiki_data(input_dir: str, output_dir: str):
    """处理Wikipedia数据"""
    print(f"Processing Wikipedia data from: {input_dir}")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "train").mkdir(exist_ok=True)
    (output_path / "eval").mkdir(exist_ok=True)
    
    cleaner = DataCleaner(min_length=50, max_length=50000)
    processor = WikipediaProcessor(cleaner)
    
    files = list(input_path.rglob("*.txt")) + list(input_path.rglob("*.jsonl"))
    
    if not files:
        print(f"Error: No text files found in {input_dir}")
        return
    
    print(f"Found {len(files)} files")
    
    all_data = []
    
    for file_path in tqdm(files, desc="Processing files"):
        try:
            processor.process_file(
                str(file_path),
                str(output_path / "temp.jsonl")
            )
            
            with open(output_path / "temp.jsonl", 'r', encoding='utf-8') as f:
                all_data.extend(f.readlines())
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if (output_path / "temp.jsonl").exists():
        os.remove(output_path / "temp.jsonl")
    
    import random
    random.shuffle(all_data)
    
    eval_size = max(100, int(len(all_data) * 0.01))
    
    with open(output_path / "train" / "wiki.jsonl", 'w', encoding='utf-8') as f:
        f.writelines(all_data[eval_size:])
    
    with open(output_path / "eval" / "wiki.jsonl", 'w', encoding='utf-8') as f:
        f.writelines(all_data[:eval_size])
    
    print(f"Done!")
    print(f"  Train: {len(all_data) - eval_size} examples")
    print(f"  Eval: {eval_size} examples")

def main():
    parser = argparse.ArgumentParser(description="数据准备脚本")
    
    parser.add_argument(
        "--source",
        type=str,
        choices=["sample", "wiki", "text"],
        default="sample",
        help="数据来源"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="输入目录（处理现有数据时使用）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="输出目录"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=1000,
        help="示例数据数量"
    )
    
    args = parser.parse_args()
    
    if args.source == "sample":
        create_sample_dataset(args.output_dir, args.num_examples)
        print("\n现在可以运行训练：")
        print(f"  python scripts/train.py --train_data {args.output_dir}/train --eval_data {args.output_dir}/eval")
    
    elif args.source == "wiki":
        if args.input is None:
            download_wikipedia()
        else:
            process_wiki_data(args.input, args.output_dir)
    
    elif args.source == "text":
        if args.input is None:
            print("请提供 --input 参数指定输入目录")
        else:
            print(f"Processing text files from: {args.input}")
            print(f"Output: {args.output_dir}")
            print("\n使用方法：")
            print("  python data/preprocess.py clean --input your_file.txt --output ./data/train/text.jsonl --format text")

if __name__ == "__main__":
    main()
