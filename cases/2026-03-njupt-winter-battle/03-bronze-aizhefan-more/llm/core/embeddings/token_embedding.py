import torch
import torch.nn as nn
from typing import Optional


class TokenEmbedding(nn.Module):
    """词嵌入层"""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        pad_token_id: int = 0,
        initializer_range: float = 0.02
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.weight, std=self.initializer_range)
        with torch.no_grad():
            self.weight[self.pad_token_id].zero_()
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]
        
        Returns:
            embeddings: [batch, seq_len, hidden_size]
        """
        embeddings = F.embedding(
            input_ids,
            self.weight,
            padding_idx=self.pad_token_id
        )
        return embeddings
    
    def linear(self, x: torch.Tensor) -> torch.Tensor:
        """
        用于LM头的线性投影（权重共享）
        
        Args:
            x: [batch, seq_len, hidden_size]
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        return F.linear(x, self.weight)


import torch.nn.functional as F
