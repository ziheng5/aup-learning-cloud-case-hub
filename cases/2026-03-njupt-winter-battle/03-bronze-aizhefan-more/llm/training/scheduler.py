import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, LinearLR, SequentialLR
from typing import Optional


class WarmupCosineScheduler(_LRScheduler):
    """
    预热 + 余弦退火学习率调度器
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = self.last_epoch
        
        if step < self.warmup_steps:
            factor = step / max(1, self.warmup_steps)
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            factor = cosine_decay * (1 - self.min_lr / self.base_lrs[0]) + self.min_lr / self.base_lrs[0]
            return [base_lr * factor for base_lr in self.base_lrs]


class WarmupLinearScheduler(_LRScheduler):
    """
    预热 + 线性衰减学习率调度器
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = self.last_epoch
        
        if step < self.warmup_steps:
            factor = step / max(1, self.warmup_steps)
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            factor = 1 - progress * (1 - self.min_lr / self.base_lrs[0])
            factor = max(factor, self.min_lr / self.base_lrs[0])
            return [base_lr * factor for base_lr in self.base_lrs]


class ConstantLRScheduler(_LRScheduler):
    """
    恒定学习率调度器（带可选预热）
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = self.last_epoch
        
        if step < self.warmup_steps:
            factor = step / max(1, self.warmup_steps)
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            return self.base_lrs


class CosineWithWarmupRestarts(_LRScheduler):
    """
    带预热重启的余弦退火调度器
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        cycle_steps: int,
        min_lr: float = 0.0,
        cycle_mult: float = 1.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.cycle_steps = cycle_steps
        self.min_lr = min_lr
        self.cycle_mult = cycle_mult
        
        self.current_cycle = 0
        self.cycle_start = warmup_steps
        self.cycle_length = cycle_steps
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = self.last_epoch
        
        if step < self.warmup_steps:
            factor = step / max(1, self.warmup_steps)
            return [base_lr * factor for base_lr in self.base_lrs]
        
        while step >= self.cycle_start + self.cycle_length:
            self.cycle_start += self.cycle_length
            self.cycle_length = int(self.cycle_length * self.cycle_mult)
            self.current_cycle += 1
        
        cycle_progress = (step - self.cycle_start) / self.cycle_length
        cosine_decay = 0.5 * (1 + math.cos(math.pi * cycle_progress))
        factor = cosine_decay * (1 - self.min_lr / self.base_lrs[0]) + self.min_lr / self.base_lrs[0]
        
        return [base_lr * factor for base_lr in self.base_lrs]


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    warmup_steps: int = 2000,
    max_steps: int = 100000,
    min_lr: float = 2e-5,
    **kwargs
) -> _LRScheduler:
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型 ("cosine", "linear", "constant", "cosine_restart")
        warmup_steps: 预热步数
        max_steps: 最大训练步数
        min_lr: 最小学习率
        **kwargs: 其他参数
    
    Returns:
        学习率调度器实例
    """
    if scheduler_type == "cosine":
        return WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr=min_lr
        )
    elif scheduler_type == "linear":
        return WarmupLinearScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr=min_lr
        )
    elif scheduler_type == "constant":
        return ConstantLRScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps
        )
    elif scheduler_type == "cosine_restart":
        cycle_steps = kwargs.get("cycle_steps", max_steps // 4)
        cycle_mult = kwargs.get("cycle_mult", 1.0)
        return CosineWithWarmupRestarts(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            cycle_steps=cycle_steps,
            min_lr=min_lr,
            cycle_mult=cycle_mult
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
