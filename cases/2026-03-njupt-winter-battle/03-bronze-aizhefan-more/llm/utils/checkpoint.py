import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    save_dir: str,
    step: int,
    epoch: int,
    loss: float,
    model_config: Optional[Any] = None,
    is_best: bool = False
) -> None:
    """
    保存检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        save_dir: 保存目录
        step: 当前步数
        epoch: 当前epoch
        loss: 当前损失
        model_config: 模型配置
        is_best: 是否是最佳模型
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model_to_save = model.module if hasattr(model, "module") else model
    
    checkpoint = {
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
    }
    
    if model_config is not None:
        checkpoint["model_config"] = model_config
    
    checkpoint_path = os.path.join(save_dir, f"checkpoint-{step}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(save_dir, "best_model.pt")
        torch.save(checkpoint, best_path)
    
    logger.info(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    加载检查点
    
    Args:
        checkpoint_path: 检查点路径
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 设备
    
    Returns:
        检查点信息字典
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    return {
        "step": checkpoint.get("step", 0),
        "epoch": checkpoint.get("epoch", 0),
        "loss": checkpoint.get("loss", float("inf")),
        "model_config": checkpoint.get("model_config", None)
    }


def load_latest_checkpoint(
    save_dir: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None
) -> Optional[Dict[str, Any]]:
    """
    加载最新检查点
    
    Args:
        save_dir: 保存目录
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 设备
    
    Returns:
        检查点信息字典，如果没有检查点则返回None
    """
    if not os.path.exists(save_dir):
        return None
    
    checkpoints = list(Path(save_dir).glob("checkpoint-*.pt"))
    
    if not checkpoints:
        return None
    
    latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split("-")[1]))
    
    return load_checkpoint(
        str(latest_checkpoint),
        model,
        optimizer,
        scheduler,
        device
    )


def export_model(
    model: nn.Module,
    save_path: str,
    model_config: Optional[Any] = None,
    export_format: str = "pt"
) -> None:
    """
    导出模型（用于推理）
    
    Args:
        model: 模型
        save_path: 保存路径
        model_config: 模型配置
        export_format: 导出格式 ("pt", "onnx")
    """
    model_to_export = model.module if hasattr(model, "module") else model
    model_to_export.eval()
    
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    
    if export_format == "pt":
        state_dict = {
            "model_state_dict": model_to_export.state_dict()
        }
        
        if model_config is not None:
            state_dict["model_config"] = model_config
        
        torch.save(state_dict, save_path)
    
    elif export_format == "onnx":
        try:
            device = next(model_to_export.parameters()).device
            dummy_input = {
                "input_ids": torch.randint(0, 1000, (1, 128), device=device),
                "attention_mask": torch.ones(1, 128, device=device)
            }
            
            torch.onnx.export(
                model_to_export,
                (dummy_input,),
                save_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "seq_length"},
                    "attention_mask": {0: "batch_size", 1: "seq_length"},
                    "logits": {0: "batch_size", 1: "seq_length"}
                }
            )
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")
            return
    
    logger.info(f"Model exported to {save_path}")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    计算模型参数数量
    
    Args:
        model: 模型
        trainable_only: 是否只计算可训练参数
    
    Returns:
        参数数量
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_summary(model: nn.Module) -> None:
    """
    打印模型摘要
    
    Args:
        model: 模型
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    print(f"Total parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e9:.2f}B)")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
