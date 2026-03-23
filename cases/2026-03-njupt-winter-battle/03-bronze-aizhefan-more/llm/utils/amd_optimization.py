import os
import torch
import gc
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

def setup_amd_env(env_vars: Optional[Dict[str, str]] = None) -> None:
    """
    设置AMD ROCm环境变量
    
    Args:
        env_vars: 可选的自定义环境变量
    """
    default_env_vars = {
        'HSA_OVERRIDE_GFX_VERSION': '11.5.1',
        'HIP_VISIBLE_DEVICES': '0',
        'ROCR_VISIBLE_DEVICES': '0',
        'AMD_SERIALIZE_KERNEL': '1',
        'TORCH_USE_HIP_DSA': '1',
        'HIP_LAUNCH_BLOCKING': '0',
        'PYTORCH_HIP_ALLOC_CONF': 'max_split_size_mb:128,expandable_segments:True',
        'HIP_FORCE_DEV_KERNARG': '1',
        'HSA_ENABLE_SDMA': '0',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
        'TORCH_CUDNN_V8_API_ENABLED': '1'
    }
    
    if env_vars:
        default_env_vars.update(env_vars)
    
    for key, value in default_env_vars.items():
        os.environ[key] = value
    
    logger.info("AMD ROCm environment variables set")


def configure_amd(
    memory_fraction: float = 0.95,
    enable_tf32: bool = True,
    enable_flash_attention: bool = True,
    compile_mode: str = "max-autotune"
) -> None:
    """
    配置AMD GPU优化
    
    Args:
        memory_fraction: 显存使用比例
        enable_tf32: 是否启用TF32
        enable_flash_attention: 是否启用FlashAttention
        compile_mode: torch.compile模式
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping AMD configuration")
        return
    
    if enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled")
    
    if enable_flash_attention and hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        logger.info("FlashAttention enabled")
    
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    logger.info(f"Memory fraction set to {memory_fraction}")
    
    if torch.__version__ >= "2.0":
        torch._dynamo.config.cache_size_limit = 128
        torch._dynamo.config.suppress_errors = True
        logger.info("torch.compile optimization enabled")
    
    logger.info("AMD GPU configuration complete")


def clear_gpu_cache() -> None:
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.debug("GPU cache cleared")


def benchmark_gpu(
    model: torch.nn.Module,
    input_shape: tuple = (1, 1024),
    num_runs: int = 100
) -> Dict[str, float]:
    """
    基准测试GPU性能
    
    Args:
        model: 模型
        input_shape: 输入形状
        num_runs: 测试次数
    
    Returns:
        性能指标字典
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    device = torch.device("cuda")
    model.eval()
    model.to(device)
    
    input_ids = torch.randint(0, 1000, input_shape, device=device)
    attention_mask = torch.ones_like(input_ids)
    
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(10):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        
        start_event.record()
        for _ in range(num_runs):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        end_event.record()
    
    torch.cuda.synchronize()
    
    elapsed_time = start_event.elapsed_time(end_event) / 1000
    avg_time = elapsed_time / num_runs
    throughput = input_shape[0] * input_shape[1] * num_runs / elapsed_time
    
    memory_allocated = torch.cuda.memory_allocated() / 1024**3
    memory_reserved = torch.cuda.memory_reserved() / 1024**3
    
    return {
        "avg_time_ms": avg_time * 1000,
        "throughput_tokens_per_sec": throughput,
        "memory_allocated_gb": memory_allocated,
        "memory_reserved_gb": memory_reserved,
        "total_time_sec": elapsed_time
    }


def print_gpu_info() -> None:
    """打印GPU信息"""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Multi-processor count: {props.multi_processor_count}")
    
    print(f"\nCurrent GPU: {torch.cuda.current_device()}")
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


def estimate_model_memory(
    num_params: int,
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    num_layers: int,
    precision: str = "bf16"
) -> Dict[str, float]:
    """
    估算模型显存使用量
    
    Args:
        num_params: 参数数量
        batch_size: 批次大小
        seq_length: 序列长度
        hidden_size: 隐藏维度
        num_layers: 层数
        precision: 精度 ("fp32", "bf16", "fp16")
    
    Returns:
        显存估算字典
    """
    bytes_per_param = {
        "fp32": 4,
        "bf16": 2,
        "fp16": 2
    }[precision]
    
    optimizer_bytes_per_param = {
        "fp32": 16,
        "bf16": 8,
        "fp16": 8
    }[precision]
    
    params_memory = num_params * bytes_per_param
    optimizer_memory = num_params * optimizer_bytes_per_param
    
    activations_per_layer = batch_size * seq_length * hidden_size * 4
    activations_memory = activations_per_layer * num_layers * bytes_per_param
    
    kv_cache_memory = 2 * batch_size * seq_length * hidden_size * num_layers * bytes_per_param
    
    total_memory = params_memory + optimizer_memory + activations_memory + kv_cache_memory
    
    return {
        "params_gb": params_memory / 1024**3,
        "optimizer_gb": optimizer_memory / 1024**3,
        "activations_gb": activations_memory / 1024**3,
        "kv_cache_gb": kv_cache_memory / 1024**3,
        "total_gb": total_memory / 1024**3
    }
