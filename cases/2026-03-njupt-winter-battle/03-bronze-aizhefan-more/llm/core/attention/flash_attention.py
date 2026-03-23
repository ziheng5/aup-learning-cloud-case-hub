import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from core.attention.rotary import RotaryPositionEmbedding, apply_rotary_pos_emb


class FlashAttention2(nn.Module):
    """
    FlashAttention 2 - 优化的自注意力实现
    支持GQA（分组查询注意力）和MQA（多查询注意力）
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        max_seq_length: int = 4096,
        rope_theta: float = 10000.0,
        rope_scaling_factor: float = 1.0,
        attention_dropout: float = 0.0,
        use_bias: bool = False,
        use_flash: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads
        self.max_seq_length = max_seq_length
        self.use_flash = use_flash
        
        self.repeat_factor = self.num_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=use_bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=use_bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        
        self.rotary_emb = RotaryPositionEmbedding(
            dim=self.head_dim,
            max_seq_length=max_seq_length,
            base=rope_theta,
            scaling_factor=rope_scaling_factor
        )
        
        self.attention_dropout = attention_dropout
        self.dropout = nn.Dropout(attention_dropout)
        
    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """重复KV头以匹配Q头数（GQA）"""
        if self.repeat_factor == 1:
            return hidden_states
        
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, self.repeat_factor, seq_len, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * self.repeat_factor, seq_len, head_dim)
    
    def _forward_flash(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = True
    ) -> torch.Tensor:
        """使用PyTorch 2.0+的scaled_dot_product_attention"""
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=q.dtype)
        
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=True
        ):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=causal
            )
        
        return attn_output
    
    def _forward_manual(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = True
    ) -> torch.Tensor:
        """手动实现的注意力（备用）"""
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        if causal:
            seq_len = q.shape[-2]
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
            attn_weights = attn_weights.masked_fill(~causal_mask, float("-inf"))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        return attn_output
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        causal: bool = True
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, 1, seq_len, seq_len] 或 None
            position_ids: [batch, seq_len] 或 None
            past_key_value: (key, value) 用于KV缓存
            use_cache: 是否返回KV缓存
            causal: 是否使用因果掩码
        
        Returns:
            attn_output: [batch, seq_len, hidden_size]
            present_key_value: (key, value) 如果use_cache=True
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(q, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=-2)
            v = torch.cat([past_key_value[1], v], dim=-2)
        
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        
        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            attn_output = self._forward_flash(q, k, v, attention_mask, causal)
        else:
            attn_output = self._forward_manual(q, k, v, attention_mask, causal)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        present_key_value = (k, v) if use_cache else None
        
        return attn_output, present_key_value


class GroupedQueryAttention(FlashAttention2):
    """
    分组查询注意力（GQA）
    当num_kv_heads=1时，退化为多查询注意力（MQA）
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int = 8,
        max_seq_length: int = 4096,
        rope_theta: float = 10000.0,
        rope_scaling_factor: float = 1.0,
        attention_dropout: float = 0.0,
        use_bias: bool = False,
        use_flash: bool = True
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_length=max_seq_length,
            rope_theta=rope_theta,
            rope_scaling_factor=rope_scaling_factor,
            attention_dropout=attention_dropout,
            use_bias=use_bias,
            use_flash=use_flash
        )


class MultiQueryAttention(GroupedQueryAttention):
    """多查询注意力（MQA）- 只有一个KV头"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_seq_length: int = 4096,
        rope_theta: float = 10000.0,
        rope_scaling_factor: float = 1.0,
        attention_dropout: float = 0.0,
        use_bias: bool = False,
        use_flash: bool = True
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=1,
            max_seq_length=max_seq_length,
            rope_theta=rope_theta,
            rope_scaling_factor=rope_scaling_factor,
            attention_dropout=attention_dropout,
            use_bias=use_bias,
            use_flash=use_flash
        )
