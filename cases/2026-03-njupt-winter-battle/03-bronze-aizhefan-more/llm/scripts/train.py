#!/usr/bin/env python3
"""
训练脚本
用法: python scripts/train.py --config configs/model_config.py
"""

import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

from configs.model_config import ModelConfig, TrainingConfig, DataConfig, AMDConfig
from core.model import LLMForCausalLM, count_parameters
from data.dataset import create_dataset
from data.dataloader import create_dataloader
from training.trainer import Trainer
from utils.amd_optimization import setup_amd_env, configure_amd, print_gpu_info

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LLM")
    
    parser.add_argument(
        "--train_data",
        type=str,
        default="./data/train",
        help="Path to training data"
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default="./data/eval",
        help="Path to evaluation data"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bert-base-chinese",
        help="Tokenizer name or path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100000,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2000,
        help="Warmup steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every X steps"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every X steps"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every X steps"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["fp32", "bf16", "fp16"],
        help="Mixed precision mode"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing"
    )
    parser.add_argument(
        "--compile_model",
        action="store_true",
        default=True,
        help="Compile model with torch.compile"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.local_rank != -1:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
    
    amd_config = AMDConfig()
    setup_amd_env(amd_config.hip_env_vars)
    configure_amd(
        memory_fraction=amd_config.memory_fraction,
        enable_tf32=amd_config.enable_tf32,
        enable_flash_attention=amd_config.enable_flash_attention,
        compile_mode=amd_config.compile_mode
    )
    
    print_gpu_info()
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id else 1,
        eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else 2
    )
    
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        mixed_precision=args.mixed_precision,
        gradient_checkpointing=args.gradient_checkpointing,
        compile_model=args.compile_model
    )
    
    logger.info(f"Creating model with {model_config.hidden_size} hidden size...")
    model = LLMForCausalLM(model_config)
    
    total_params = count_parameters(model)
    logger.info(f"Model created with {total_params / 1e9:.2f}B parameters")
    
    logger.info("Loading datasets...")
    train_dataset = create_dataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        max_seq_length=model_config.max_seq_length,
        dataset_type="text"
    )
    logger.info(f"Train dataset size: {len(train_dataset)}")
    
    eval_dataset = None
    if os.path.exists(args.eval_data):
        eval_dataset = create_dataset(
            data_path=args.eval_data,
            tokenizer=tokenizer,
            max_seq_length=model_config.max_seq_length,
            dataset_type="text"
        )
        logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    logger.info("Creating dataloaders...")
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        distributed=args.local_rank != -1,
        rank=args.local_rank,
        world_size=torch.distributed.get_world_size() if args.local_rank != -1 else 1,
        tokenizer=tokenizer
    )
    
    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = create_dataloader(
            eval_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
            tokenizer=tokenizer
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_config,
        model_config=model_config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        device=device,
        local_rank=args.local_rank,
        world_size=torch.distributed.get_world_size() if args.local_rank != -1 else 1
    )
    
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
