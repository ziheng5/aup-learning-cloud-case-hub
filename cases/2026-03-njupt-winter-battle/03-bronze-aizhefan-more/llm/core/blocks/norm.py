import torch
import torch.nn as nn
from typing import Optional


class RMSNorm(nn.Module):
    """
    RMS归一化
    RMSNorm(x) = x * rms(x) * g
    其中rms(x) = sqrt(mean(x^2))
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        
        Returns:
            normalized_states: [batch, seq_len, hidden_size]
        """
        output = self._norm(hidden_states.float()).type_as(hidden_states)
        return output * self.weight


class LayerNorm(nn.Module):
    """标准LayerNorm - 备用"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6, bias: bool = True):
        super().__init__()
        
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mean = hidden_states.mean(-1, keepdim=True)
        var = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) * torch.rsqrt(var + self.eps)
        
        output = hidden_states * self.weight
        if self.bias is not None:
            output = output + self.bias
        
        return output
