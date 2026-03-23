from dataclasses import dataclass, field
from typing import Optional, List
import torch

@dataclass
class ModelConfig:
    """3B参数模型配置 - 面向日常对话的Decoder-only Transformer"""
    
    vocab_size: int = 65536
    hidden_size: int = 3072
    num_layers: int = 32
    num_heads: int = 24
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 8192
    max_seq_length: int = 4096
    
    norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_scaling: Optional[dict] = field(default_factory=lambda: {
        "type": "linear",
        "factor": 4.0
    })
    
    hidden_act: str = "swiglu"
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    
    tie_word_embeddings: bool = True
    use_cache: bool = True
    use_flash_attention: bool = True
    
    initializer_range: float = 0.02
    initializer_std: float = 0.02
    
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    def __post_init__(self):
        assert self.hidden_size % self.num_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
        self.head_dim = self.hidden_size // self.num_heads

@dataclass
class TrainingConfig:
    """训练配置"""
    
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    effective_batch_size: int = field(init=False)
    
    learning_rate: float = 2e-4
    min_lr: float = 2e-5
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    
    max_steps: int = 100000
    warmup_steps: int = 2000
    
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 1000
    
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    
    mixed_precision: str = "bf16"
    compile_model: bool = True
    
    output_dir: str = "./checkpoints"
    logging_dir: str = "./logs"
    
    def __post_init__(self):
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps

@dataclass
class DataConfig:
    """数据配置"""
    
    train_data_path: str = "./data/train"
    eval_data_path: str = "./data/eval"
    tokenizer_path: str = "./tokenizer"
    
    max_seq_length: int = 4096
    shuffle_buffer_size: int = 10000
    
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

@dataclass
class InferenceConfig:
    """推理配置"""
    
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    
    do_sample: bool = True
    use_cache: bool = True
    
    pad_token_id: int = 0
    eos_token_id: int = 2

@dataclass
class AMDConfig:
    """AMD ROCm优化配置"""
    
    use_hip: bool = True
    gfx_version: str = "11.5.1"
    
    enable_flash_attention: bool = True
    enable_torch_compile: bool = True
    compile_mode: str = "max-autotune"
    
    memory_fraction: float = 0.95
    enable_tf32: bool = True
    
    hip_env_vars: dict = field(default_factory=lambda: {
        'HSA_OVERRIDE_GFX_VERSION': '11.5.1',
        'HIP_VISIBLE_DEVICES': '0',
        'ROCR_VISIBLE_DEVICES': '0',
        'AMD_SERIALIZE_KERNEL': '1',
        'TORCH_USE_HIP_DSA': '1',
        'HIP_LAUNCH_BLOCKING': '0',
        'PYTORCH_HIP_ALLOC_CONF': 'max_split_size_mb:128,expandable_segments:True',
        'HIP_FORCE_DEV_KERNARG': '1',
        'HSA_ENABLE_SDMA': '0'
    })
