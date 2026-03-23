import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU前馈网络
    FFN(x) = max(0, xW1 + b1) * (xW3 + b3)W2 + b2
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_dropout: float = 0.0,
        use_bias: bool = False
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=use_bias)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        
        self.dropout = nn.Dropout(hidden_dropout)
        self.act_fn = nn.SiLU()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        gate = self.act_fn(self.w1(hidden_states))
        up = self.w3(hidden_states)
        intermediate_states = gate * up
        intermediate_states = self.dropout(intermediate_states)
        output = self.w2(intermediate_states)
        
        return output


class GatedFeedForward(nn.Module):
    """门控前馈网络 - 标准实现"""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_dropout: float = 0.0,
        use_bias: bool = False,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=use_bias)
        
        self.dropout = nn.Dropout(hidden_dropout)
        
        if activation == "gelu":
            self.act_fn = nn.GELU()
        elif activation == "relu":
            self.act_fn = nn.ReLU()
        elif activation == "silu":
            self.act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.up_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        return hidden_states


class MoEFeedForward(nn.Module):
    """
    混合专家模型（MoE）- 可选用于未来扩展
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        hidden_dropout: float = 0.0,
        use_bias: bool = False
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        self.experts = nn.ModuleList([
            SwiGLUFeedForward(hidden_size, intermediate_size, hidden_dropout, use_bias)
            for _ in range(num_experts)
        ])
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        gate_logits = self.gate(hidden_states)
        gates = F.softmax(gate_logits, dim=-1)
        
        top_k_gates, top_k_indices = gates.topk(self.top_k, dim=-1)
        top_k_gates = top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)
        
        flat_hidden = hidden_states.reshape(-1, hidden_size)
        flat_indices = top_k_indices.reshape(-1, self.top_k)
        flat_gates = top_k_gates.reshape(-1, self.top_k)
        
        output = torch.zeros_like(flat_hidden)
        
        for expert_idx, expert in enumerate(self.experts):
            expert_mask = (flat_indices == expert_idx).any(dim=-1)
            
            if expert_mask.any():
                expert_input = flat_hidden[expert_mask]
                expert_output = expert(expert_input)
                
                gate_positions = (flat_indices[expert_mask] == expert_idx).nonzero(as_tuple=True)
                expert_gates = flat_gates[expert_mask][gate_positions[0], gate_positions[1]]
                expert_gates = expert_gates.unsqueeze(-1)
                
                output[expert_mask] += expert_output * expert_gates
        
        return output.reshape(batch_size, seq_len, hidden_size)
