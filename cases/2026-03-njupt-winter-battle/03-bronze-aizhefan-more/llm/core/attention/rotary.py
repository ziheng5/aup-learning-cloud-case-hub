import torch
import torch.nn as nn
from typing import Optional, Tuple


class RotaryPositionEmbedding(nn.Module):
    """RoPE旋转位置编码 - 支持线性缩放"""
    
    def __init__(
        self,
        dim: int,
        max_seq_length: int = 4096,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.base = base
        self.scaling_factor = scaling_factor
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        self._set_cos_sin_cache(max_seq_length, device)
    
    def _set_cos_sin_cache(self, seq_len: int, device: Optional[torch.device] = None):
        """预计算cos和sin缓存"""
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor
        
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
    
    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, num_heads, seq_len, head_dim]
            seq_len: 当前序列长度
        
        Returns:
            cos, sin: [1, 1, seq_len, head_dim]
        """
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device)
        
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype)
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """旋转向量的一半"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对Q和K应用RoPE
    
    Args:
        q: [batch, num_heads, seq_len, head_dim]
        k: [batch, num_kv_heads, seq_len, head_dim]
        cos, sin: [1, 1, seq_len, head_dim]
        position_ids: 可选的位置ID
    
    Returns:
        q_embed, k_embed: 应用RoPE后的Q和K
    """
    if position_ids is not None:
        cos = cos.squeeze(1).squeeze(0)
        sin = sin.squeeze(1).squeeze(0)
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class LinearScaledRoPE(RotaryPositionEmbedding):
    """线性缩放的RoPE - 支持更长序列"""
    
    def __init__(
        self,
        dim: int,
        max_seq_length: int = 4096,
        base: float = 10000.0,
        scaling_factor: float = 4.0,
        device: Optional[torch.device] = None
    ):
        super().__init__(dim, max_seq_length, base, scaling_factor, device)
        self.scaling_factor = scaling_factor
    
    def _set_cos_sin_cache(self, seq_len: int, device: Optional[torch.device] = None):
        """使用线性缩放计算缓存"""
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor
        
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
