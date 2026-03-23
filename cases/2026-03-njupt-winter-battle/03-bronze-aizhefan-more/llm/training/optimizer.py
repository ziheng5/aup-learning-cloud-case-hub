import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from typing import List, Dict, Optional, Tuple
import math


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.1,
    no_decay_params: Tuple[str, ...] = ("bias", "norm")
) -> List[Dict]:
    """
    获取参数组（用于权重衰减）
    
    Args:
        model: 模型
        weight_decay: 权重衰减系数
        no_decay_params: 不使用权重衰减的参数名
    
    Returns:
        参数组列表
    """
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay_params)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay_params)
            ],
            "weight_decay": 0.0,
        },
    ]
    
    return optimizer_grouped_parameters


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    no_decay_params: Tuple[str, ...] = ("bias", "norm")
) -> Optimizer:
    """
    创建AdamW优化器
    
    Args:
        model: 模型
        learning_rate: 学习率
        weight_decay: 权重衰减
        betas: AdamW beta参数
        eps: AdamW epsilon
        no_decay_params: 不使用权重衰减的参数名
    
    Returns:
        AdamW优化器实例
    """
    optimizer_grouped_parameters = get_parameter_groups(
        model, weight_decay, no_decay_params
    )
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=betas,
        eps=eps
    )
    
    return optimizer


class Lion(Optimizer):
    """
    Lion优化器
    论文：Symbolic Discovery of Optimization Algorithms
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                
                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]
                
                state["step"] += 1
                
                weight_decay = group["weight_decay"]
                lr = group["lr"]
                
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-lr)
                
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        
        return loss


class Adafactor(Optimizer):
    """
    Adafactor优化器
    适用于大模型训练，内存效率更高
    """
    
    def __init__(
        self,
        params,
        lr: Optional[float] = None,
        eps: Tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1: Optional[float] = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = True,
        warmup_init: bool = False
    ):
        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init
        )
        super().__init__(params, defaults)
    
    def _get_lr(self, param_group, param_state):
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = 1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz
    
    def _get_options(self, param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment
    
    def _rms(self, x):
        return x.norm(2) / math.sqrt(x.numel())
    
    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (
            exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)
        ).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.rsqrt().unsqueeze(-2)
        return torch.mm(r_factor, c_factor)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")
                
                state = self.state[p]
                grad_shape = grad.shape
                
                factored, use_first_moment = self._get_options(group, grad_shape)
                
                if len(state) == 0:
                    state["step"] = 0
                    if use_first_moment:
                        state["exp_avg"] = torch.zeros_like(p)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1], device=grad.device)
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:], device=grad.device)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    
                    state["RMS"] = 0
                else:
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad.device)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad.device)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad.device)
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(grad.device)
                
                state["step"] += 1
                state["RMS"] = self._rms(p)
                
                lr = self._get_lr(group, state)
                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad ** 2) + group["eps"][0]
                
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    
                    exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=(1.0 - beta2t))
                    exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=(1.0 - beta2t))
                    
                    approx_exp_avg_sq = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update = approx_exp_avg_sq.rsqrt() * grad
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
                    update = exp_avg_sq.rsqrt() * grad
                
                update = update / update.norm(2).clamp(min=1e-6) * group["clip_threshold"]
                
                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=(1 - group["beta1"]))
                    update = exp_avg
                
                if group["weight_decay"] != 0:
                    p.add_(p, alpha=(-group["weight_decay"] * lr))
                
                p.add_(update, alpha=-lr)
        
        return loss
