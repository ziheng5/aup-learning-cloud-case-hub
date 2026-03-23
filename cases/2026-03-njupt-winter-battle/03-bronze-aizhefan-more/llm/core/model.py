import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union
import os

from core.embeddings.token_embedding import TokenEmbedding
from core.blocks.transformer import TransformerStack
from core.blocks.norm import RMSNorm
from configs.model_config import ModelConfig


class LLMModel(nn.Module):
    """
    3B参数Decoder-only Transformer模型
    面向日常对话的语言模型
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers
        
        self.embed_tokens = TokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            pad_token_id=config.pad_token_id,
            initializer_range=config.initializer_range
        )
        
        self.layers = TransformerStack(
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            intermediate_size=config.intermediate_size,
            max_seq_length=config.max_seq_length,
            rope_theta=config.rope_theta,
            rope_scaling_factor=config.rope_scaling.get("factor", 1.0) if config.rope_scaling else 1.0,
            norm_eps=config.norm_eps,
            attention_dropout=config.attention_dropout,
            hidden_dropout=config.hidden_dropout,
            use_bias=False,
            use_flash_attention=config.use_flash_attention,
            use_gradient_checkpointing=config.gradient_checkpointing if hasattr(config, 'gradient_checkpointing') else False
        )
        
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        
        self.gradient_checkpointing = False
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def enable_gradient_checkpointing(self):
        """启用梯度检查点"""
        self.gradient_checkpointing = True
        self.layers.use_gradient_checkpointing = True
        for layer in self.layers.layers:
            layer.use_gradient_checkpointing = True
    
    def disable_gradient_checkpointing(self):
        """禁用梯度检查点"""
        self.gradient_checkpointing = False
        self.layers.use_gradient_checkpointing = False
        for layer in self.layers.layers:
            layer.use_gradient_checkpointing = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        return_dict: bool = True
    ) -> Union[Dict, Tuple]:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] 或 None
            position_ids: [batch, seq_len] 或 None
            past_key_values: KV缓存列表
            use_cache: 是否使用KV缓存
        
        Returns:
            last_hidden_state: [batch, seq_len, hidden_size]
            past_key_values: KV缓存（如果use_cache=True）
        """
        batch_size, seq_length = input_ids.shape
        
        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * self.num_layers
        else:
            past_length = past_key_values[0][0].shape[-2]
        
        if position_ids is None:
            position_ids = torch.arange(
                past_length, seq_length + past_length,
                dtype=torch.long,
                device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        inputs_embeds = self.embed_tokens(input_ids)
        
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            
            attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(inputs_embeds.dtype).min
        
        outputs = self.layers(
            hidden_states=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict
        )
        
        if return_dict:
            hidden_states = outputs["last_hidden_state"]
            past_key_values = outputs["past_key_values"]
        else:
            hidden_states, past_key_values = outputs
        
        hidden_states = self.norm(hidden_states)
        
        if return_dict:
            return {
                "last_hidden_state": hidden_states,
                "past_key_values": past_key_values
            }
        
        return hidden_states, past_key_values


class LLMForCausalLM(nn.Module):
    """
    用于因果语言建模的3B模型
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        self.model = LLMModel(config)
        
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        if self.lm_head is not None:
            nn.init.normal_(self.lm_head.weight, std=self.config.initializer_std)
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        if self.config.tie_word_embeddings:
            self.model.embed_tokens = new_embeddings
        else:
            self.lm_head = new_embeddings
    
    def tie_weights(self):
        """绑定输入输出嵌入权重"""
        if self.config.tie_word_embeddings:
            self.lm_head = None
    
    def enable_gradient_checkpointing(self):
        self.model.enable_gradient_checkpointing()
    
    def disable_gradient_checkpointing(self):
        self.model.disable_gradient_checkpointing()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        return_dict: bool = True
    ) -> Union[Dict, Tuple]:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            position_ids: [batch, seq_len]
            past_key_values: KV缓存
            labels: [batch, seq_len] - 用于计算损失
            use_cache: 是否使用KV缓存
        
        Returns:
            loss: 交叉熵损失（如果提供labels）
            logits: [batch, seq_len, vocab_size]
            past_key_values: KV缓存
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict
        )
        
        if return_dict:
            hidden_states = outputs["last_hidden_state"]
            past_key_values = outputs["past_key_values"]
        else:
            hidden_states = outputs[0]
            past_key_values = outputs[1] if len(outputs) > 1 else None
        
        if self.config.tie_word_embeddings:
            logits = F.linear(hidden_states, self.model.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        if return_dict:
            return {
                "loss": loss,
                "logits": logits,
                "past_key_values": past_key_values
            }
        
        output = (logits,)
        if past_key_values is not None:
            output = output + (past_key_values,)
        if loss is not None:
            output = (loss,) + output
        
        return output
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[list] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict:
        """准备生成输入"""
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            input_ids = input_ids[:, past_length:]
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": True
        }
    
    @staticmethod
    def _reorder_cache(past_key_values: list, beam_idx: torch.Tensor) -> list:
        """重新排序缓存（用于束搜索）"""
        return tuple(
            tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
            for layer_past in past_key_values
        )
    
    def save_pretrained(self, save_directory: str):
        """保存模型"""
        os.makedirs(save_directory, exist_ok=True)
        
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        config_path = os.path.join(save_directory, "config.pt")
        torch.save(self.config, config_path)
    
    @classmethod
    def from_pretrained(cls, load_directory: str) -> "LLMForCausalLM":
        """加载模型"""
        config_path = os.path.join(load_directory, "config.pt")
        config = torch.load(config_path, weights_only=False)
        
        model = cls(config)
        
        model_path = os.path.join(load_directory, "pytorch_model.bin")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        
        return model


def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_memory_usage(config: ModelConfig, batch_size: int, seq_length: int) -> Dict:
    """
    估算显存使用量
    返回：参数量、激活值、KV缓存等的显存估算
    """
    params = count_parameters(LLMForCausalLM(config))
    
    params_memory = params * 2  # BF16
    
    activations_memory = (
        batch_size * seq_length * config.hidden_size *
        config.num_layers * 4 * 2  # 4个激活值/层，BF16
    )
    
    kv_cache_memory = (
        2 * batch_size * config.num_layers * seq_length *
        config.hidden_size * 2  # BF16
    )
    
    optimizer_memory = params * 8  # AdamW状态
    
    return {
        "parameters": params,
        "params_memory_gb": params_memory / (1024**3),
        "activations_memory_gb": activations_memory / (1024**3),
        "kv_cache_memory_gb": kv_cache_memory / (1024**3),
        "optimizer_memory_gb": optimizer_memory / (1024**3),
        "total_training_memory_gb": (
            params_memory + activations_memory + kv_cache_memory + optimizer_memory
        ) / (1024**3)
    }
