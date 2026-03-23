#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wikipedia data extraction using gensim
"""

import os
import json
import argparse
import re
from pathlib import Path
from tqdm import tqdm
import random

def clean_text(text: str) -> str:
    """Clean Wikipedia text"""
    text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\[\[.*?\]\]', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'={2,}.*?={2,}', '', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def process_wikipedia_dump(
    input_path: str,
    output_dir: str,
    min_length: int = 100,
    max_length: int = 10000,
    eval_ratio: float = 0.001
):
    """
    Process Wikipedia XML.bz2 file
    
    Args:
        input_path: Wikipedia dump file path (.xml.bz2)
        output_dir: Output directory
        min_length: Minimum text length
        max_length: Maximum text length
        eval_ratio: Evaluation set ratio
    """
    from gensim.corpora.wikicorpus import extract_pages, filter_wiki
    
    input_path = Path(input_path)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: File not found {input_path}")
        return
    
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "train").mkdir(exist_ok=True)
    (output_path / "eval").mkdir(exist_ok=True)
    
    print(f"Processing file: {input_path}")
    print("This may take several hours...")
    
    all_docs = []
    total_chars = 0
    
    try:
        wiki = extract_pages(str(input_path))
        
        for title, text, pageid in tqdm(wiki, desc="Extracting pages"):
            text = filter_wiki(text)
            text = clean_text(text)
            
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
            
            if len(all_docs) % 10000 == 0 and len(all_docs) > 0:
                print(f"Extracted {len(all_docs):,} docs, {total_chars/1024/1024/1024:.2f} GB")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user, saving current data...")
    except Exception as e:
        print(f"\nError: {e}")
        print("Saving current data...")
    
    print(f"\nProcessing complete!")
    print(f"  Total docs: {len(all_docs):,}")
    print(f"  Total chars: {total_chars:,}")
    print(f"  Estimated size: {total_chars / 1024 / 1024 / 1024:.2f} GB")
    
    if len(all_docs) == 0:
        print("Error: No data extracted")
        return
    
    print("\nShuffling data...")
    random.shuffle(all_docs)
    
    eval_size = max(1000, int(len(all_docs) * eval_ratio))
    train_docs = all_docs[eval_size:]
    eval_docs = all_docs[:eval_size]
    
    print(f"\nSplitting dataset:")
    print(f"  Train: {len(train_docs):,}")
    print(f"  Eval: {len(eval_docs):,}")
    
    train_file = output_path / "train" / "wikipedia_zh.jsonl"
    eval_file = output_path / "eval" / "wikipedia_zh.jsonl"
    
    print(f"\nWriting train set: {train_file}")
    with open(train_file, 'w', encoding='utf-8') as f:
        for doc in tqdm(train_docs, desc="Writing train"):
            f.write(json.dumps({"text": doc}, ensure_ascii=False) + '\n')
    
    print(f"Writing eval set: {eval_file}")
    with open(eval_file, 'w', encoding='utf-8') as f:
        for doc in tqdm(eval_docs, desc="Writing eval"):
            f.write(json.dumps({"text": doc}, ensure_ascii=False) + '\n')
    
    train_size = train_file.stat().st_size / 1024 / 1024 / 1024
    eval_size = eval_file.stat().st_size / 1024 / 1024 / 1024
    
    print(f"\nFinal output:")
    print(f"  Train: {train_file} ({train_size:.2f} GB)")
    print(f"  Eval: {eval_file} ({eval_size:.2f} GB)")
    print(f"  Total: {train_size + eval_size:.2f} GB")
    
    print("\n" + "="*60)
    print("Data preparation complete! Now start training:")
    print(f"python scripts/train.py --train_data {output_path}/train --eval_data {output_path}/eval")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Extract Wikipedia data using gensim")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Wikipedia dump file path (.xml.bz2)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory"
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=100,
        help="Minimum text length"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=10000,
        help="Maximum text length"
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.001,
        help="Evaluation set ratio"
    )
    
    args = parser.parse_args()
    
    process_wikipedia_dump(
        input_path=args.input,
        output_dir=args.output_dir,
        min_length=args.min_length,
        max_length=args.max_length,
        eval_ratio=args.eval_ratio
    )

if __name__ == "__main__":
    main()
