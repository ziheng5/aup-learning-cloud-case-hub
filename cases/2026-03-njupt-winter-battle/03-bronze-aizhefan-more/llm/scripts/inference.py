#!/usr/bin/env python3
"""
推理脚本
用法: python scripts/inference.py --model_path ./checkpoints/final --prompt "你好"
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

from configs.model_config import ModelConfig, InferenceConfig, AMDConfig
from core.model import LLMForCausalLM
from inference.generator import SamplingGenerator, BeamSearchGenerator
from utils.amd_optimization import setup_amd_env, configure_amd

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with LLM")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bert-base-chinese",
        help="Tokenizer name or path"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="你好，请介绍一下自己。",
        help="Input prompt"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p for nucleus sampling"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k for sampling"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="Repetition penalty"
    )
    parser.add_argument(
        "--use_beam_search",
        action="store_true",
        default=False,
        help="Use beam search instead of sampling"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Number of beams for beam search"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Interactive mode"
    )
    
    return parser.parse_args()


def load_model(model_path: str, device: torch.device) -> LLMForCausalLM:
    """加载模型"""
    config_path = os.path.join(model_path, "config.pt")
    model_path_bin = os.path.join(model_path, "pytorch_model.bin")
    
    if os.path.exists(config_path):
        config = torch.load(config_path, weights_only=False)
        model = LLMForCausalLM(config)
    else:
        config = ModelConfig()
        model = LLMForCausalLM(config)
    
    if os.path.exists(model_path_bin):
        state_dict = torch.load(model_path_bin, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Model weights not found at {model_path_bin}")
        print("Using random weights for testing")
    
    model.eval()
    model.to(device)
    
    return model


def main():
    args = parse_args()
    
    setup_amd_env()
    configure_amd()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path, device)
    
    inference_config = InferenceConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id or 2
    )
    
    if args.use_beam_search:
        generator = BeamSearchGenerator(
            model=model,
            tokenizer=tokenizer,
            config=inference_config,
            device=device,
            num_beams=args.num_beams
        )
    else:
        generator = SamplingGenerator(
            model=model,
            tokenizer=tokenizer,
            config=inference_config,
            device=device
        )
    
    if args.interactive:
        print("\n" + "="*50)
        print("交互式对话模式（输入 'quit' 退出）")
        print("="*50 + "\n")
        
        while True:
            prompt = input("用户: ")
            if prompt.lower() in ["quit", "exit", "退出"]:
                break
            
            with torch.no_grad():
                response = generator.generate(prompt)
            
            print(f"\n助手: {response[len(prompt):].strip()}\n")
    else:
        print(f"\n提示: {args.prompt}\n")
        
        with torch.no_grad():
            response = generator.generate(args.prompt)
        
        generated_text = response[len(args.prompt):].strip()
        print(f"生成: {generated_text}\n")


if __name__ == "__main__":
    main()
