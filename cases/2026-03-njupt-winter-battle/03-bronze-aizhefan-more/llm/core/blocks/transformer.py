import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import torch.utils.checkpoint as checkpoint

from core.attention.flash_attention import GroupedQueryAttention
from core.feedforward.swiglu import SwiGLUFeedForward
from core.blocks.norm import RMSNorm


class TransformerBlock(nn.Module):
    """
    Transformer Decoder块
    使用Pre-LN和RMSNorm
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        max_seq_length: int = 4096,
        rope_theta: float = 10000.0,
        rope_scaling_factor: float = 1.0,
        norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        use_bias: bool = False,
        use_flash_attention: bool = True,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        self.input_layernorm = RMSNorm(hidden_size, eps=norm_eps)
        
        self.self_attn = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_length=max_seq_length,
            rope_theta=rope_theta,
            rope_scaling_factor=rope_scaling_factor,
            attention_dropout=attention_dropout,
            use_bias=use_bias,
            use_flash=use_flash_attention
        )
        
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=norm_eps)
        
        self.mlp = SwiGLUFeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_dropout=hidden_dropout,
            use_bias=use_bias
        )
    
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
            hidden_states: [batch, seq_len, hidden_size]
            present_key_value: (key, value) 如果use_cache=True
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        attn_output, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            causal=causal
        )
        
        hidden_states = residual + attn_output
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value


class TransformerStack(nn.Module):
    """
    Transformer层堆叠
    """
    
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        max_seq_length: int = 4096,
        rope_theta: float = 10000.0,
        rope_scaling_factor: float = 1.0,
        norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        use_bias: bool = False,
        use_flash_attention: bool = True,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                intermediate_size=intermediate_size,
                max_seq_length=max_seq_length,
                rope_theta=rope_theta,
                rope_scaling_factor=rope_scaling_factor,
                norm_eps=norm_eps,
                attention_dropout=attention_dropout,
                hidden_dropout=hidden_dropout,
                use_bias=use_bias,
                use_flash_attention=use_flash_attention,
                use_gradient_checkpointing=use_gradient_checkpointing
            )
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        return_dict: bool = True
    ):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, 1, seq_len, seq_len] 或 None
            position_ids: [batch, seq_len] 或 None
            past_key_values: 每层的KV缓存列表
            use_cache: 是否返回KV缓存
        
        Returns:
            hidden_states: [batch, seq_len, hidden_size]
            next_cache: KV缓存列表（如果use_cache=True）
        """
        next_cache = [] if use_cache else None
        
        for idx, layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            if self.use_gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, use_cache, True)
                    return custom_forward
                
                layer_outputs = checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value
                )
            else:
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    causal=True
                )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_cache.append(layer_outputs[1])
        
        if return_dict:
            return {"last_hidden_state": hidden_states, "past_key_values": next_cache}
        
        return hidden_states, next_cache
