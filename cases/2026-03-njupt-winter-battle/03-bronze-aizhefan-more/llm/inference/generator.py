import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Union
import warnings

from configs.model_config import InferenceConfig


class Generator:
    """
    文本生成器基类
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Optional[InferenceConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or InferenceConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.eval()
        self.model.to(self.device)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            **kwargs: 其他生成参数
        
        Returns:
            生成的文本
        """
        raise NotImplementedError
    
    def _prepare_inputs(self, prompt: str) -> Dict[str, torch.Tensor]:
        """准备输入"""
        encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        return {
            "input_ids": encoding["input_ids"].to(self.device),
            "attention_mask": encoding["attention_mask"].to(self.device)
        }
    
    def _decode(self, token_ids: torch.Tensor) -> str:
        """解码token"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


class SamplingGenerator(Generator):
    """
    采样生成器
    支持top-k、top-p、温度采样
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Optional[InferenceConfig] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__(model, tokenizer, config, device)
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        使用采样策略生成文本
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            temperature: 温度
            top_p: 核采样概率
            top_k: top-k采样
            repetition_penalty: 重复惩罚
            **kwargs: 其他参数
        
        Returns:
            生成的文本
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k
        repetition_penalty = repetition_penalty or self.config.repetition_penalty
        
        inputs = self._prepare_inputs(prompt)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        past_key_values = None
        generated_tokens = []
        
        for _ in range(max_new_tokens):
            if past_key_values is not None:
                model_inputs = {
                    "input_ids": input_ids[:, -1:],
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values,
                    "use_cache": True
                }
            else:
                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "use_cache": True
                }
            
            outputs = self.model(**model_inputs)
            
            if isinstance(outputs, dict):
                logits = outputs["logits"]
                past_key_values = outputs.get("past_key_values")
            else:
                logits = outputs[0]
                past_key_values = outputs[1] if len(outputs) > 1 else None
            
            next_token_logits = logits[:, -1, :]
            
            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    if next_token_logits[0, token_id] < 0:
                        next_token_logits[0, token_id] *= repetition_penalty
                    else:
                        next_token_logits[0, token_id] /= repetition_penalty
            
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            next_token_logits = self._top_k_top_p_filtering(
                next_token_logits,
                top_k=top_k,
                top_p=top_p
            )
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == self.config.eos_token_id:
                break
            
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
        
        full_response = prompt + self._decode(torch.tensor(generated_tokens))
        return full_response
    
    @staticmethod
    def _top_k_top_p_filtering(
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = float("-inf"),
        min_tokens_to_keep: int = 1
    ) -> torch.Tensor:
        """
        Top-K和Top-P过滤
        
        Args:
            logits: 模型logits
            top_k: top-k值
            top_p: top-p值
            filter_value: 过滤值
            min_tokens_to_keep: 最小保留token数
        
        Returns:
            过滤后的logits
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            
            if min_tokens_to_keep > 1:
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = filter_value
        
        return logits


class BeamSearchGenerator(Generator):
    """
    束搜索生成器
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Optional[InferenceConfig] = None,
        device: Optional[torch.device] = None,
        num_beams: int = 4,
        length_penalty: float = 1.0
    ):
        super().__init__(model, tokenizer, config, device)
        self.num_beams = num_beams
        self.length_penalty = length_penalty
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        num_beams: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        使用束搜索生成文本
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            num_beams: 束数
            **kwargs: 其他参数
        
        Returns:
            生成的文本
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        num_beams = num_beams or self.num_beams
        
        inputs = self._prepare_inputs(prompt)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        batch_size = input_ids.shape[0]
        cur_len = input_ids.shape[1]
        
        input_ids = input_ids.repeat_interleave(num_beams, dim=0)
        attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)
        
        beam_scores = torch.zeros(batch_size, num_beams, device=self.device)
        beam_scores[:, 1:] = float("-inf")
        beam_scores = beam_scores.view(-1)
        
        done = torch.zeros(batch_size * num_beams, dtype=torch.bool, device=self.device)
        
        past_key_values = None
        
        for step in range(max_new_tokens):
            model_inputs = {
                "input_ids": input_ids[:, -1:] if step > 0 else input_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": True
            }
            
            outputs = self.model(**model_inputs)
            
            if isinstance(outputs, dict):
                logits = outputs["logits"]
                past_key_values = outputs.get("past_key_values")
            else:
                logits = outputs[0]
                past_key_values = outputs[1] if len(outputs) > 1 else None
            
            next_token_logits = logits[:, -1, :]
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            
            next_token_scores = next_token_scores * (1 - done.float()).unsqueeze(1)
            next_token_scores[done, self.config.eos_token_id] = 0
            next_token_scores[done, self.config.eos_token_id + 1:] = float("-inf")
            
            length_penalty = ((cur_len + step + 1) / cur_len) ** self.length_penalty
            next_token_scores = next_token_scores / length_penalty
            
            next_token_scores = next_token_scores + beam_scores.unsqueeze(1)
            
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            
            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size
            
            beam_outputs = self._update_beam(
                next_token_scores,
                next_tokens,
                next_indices,
                batch_size,
                num_beams,
                done,
                step,
                max_new_tokens
            )
            
            if beam_outputs["done"].all():
                break
            
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]
            
            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([
                attention_mask[beam_idx, :],
                torch.ones_like(beam_next_tokens).unsqueeze(-1)
            ], dim=-1)
            
            done = beam_outputs["done"]
        
        best_beam_idx = beam_scores.view(batch_size, num_beams).argmax(-1)
        best_beam_idx = best_beam_idx + torch.arange(batch_size, device=self.device) * num_beams
        
        best_sequence = input_ids[best_beam_idx[0]]
        return self._decode(best_sequence)
    
    def _update_beam(
        self,
        next_token_scores: torch.Tensor,
        next_tokens: torch.Tensor,
        next_indices: torch.Tensor,
        batch_size: int,
        num_beams: int,
        done: torch.Tensor,
        cur_step: int,
        max_steps: int
    ) -> Dict[str, torch.Tensor]:
        """更新束状态"""
        next_beam_scores = torch.zeros(batch_size, num_beams, device=self.device)
        next_beam_tokens = torch.zeros(batch_size, num_beams, dtype=torch.long, device=self.device)
        next_beam_indices = torch.zeros(batch_size, num_beams, dtype=torch.long, device=self.device)
        
        next_done = done.clone()
        
        for batch_idx in range(batch_size):
            beam_idx = 0
            for token_idx, (score, token, beam_index) in enumerate(zip(
                next_token_scores[batch_idx],
                next_tokens[batch_idx],
                next_indices[batch_idx]
            )):
                if beam_idx >= num_beams:
                    break
                
                global_beam_idx = batch_idx * num_beams + beam_index
                
                if token == self.config.eos_token_id and cur_step < max_steps - 1:
                    if not done[global_beam_idx]:
                        pass
                    continue
                
                next_beam_scores[batch_idx, beam_idx] = score
                next_beam_tokens[batch_idx, beam_idx] = token
                next_beam_indices[batch_idx, beam_idx] = global_beam_idx
                beam_idx += 1
        
        return {
            "next_beam_scores": next_beam_scores.view(-1),
            "next_beam_tokens": next_beam_tokens.view(-1),
            "next_beam_indices": next_beam_indices.view(-1),
            "done": next_done
        }


class ContrastiveSearchGenerator(Generator):
    """
    对比搜索生成器
    论文：A Contrastive Framework for Neural Text Generation
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Optional[InferenceConfig] = None,
        device: Optional[torch.device] = None,
        penalty_alpha: float = 0.6,
        top_k: int = 4
    ):
        super().__init__(model, tokenizer, config, device)
        self.penalty_alpha = penalty_alpha
        self.top_k = top_k
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        penalty_alpha: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        使用对比搜索生成
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            penalty_alpha: 惩罚系数
            top_k: top-k候选数
            **kwargs: 其他参数
        
        Returns:
            生成的文本
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        penalty_alpha = penalty_alpha or self.penalty_alpha
        top_k = top_k or self.top_k
        
        inputs = self._prepare_inputs(prompt)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        past_key_values = None
        generated_tokens = []
        
        for _ in range(max_new_tokens):
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": True
            }
            
            outputs = self.model(**model_inputs)
            
            if isinstance(outputs, dict):
                logits = outputs["logits"]
                past_key_values = outputs.get("past_key_values")
            else:
                logits = outputs[0]
            
            next_token_logits = logits[:, -1, :]
            scores = F.softmax(next_token_logits, dim=-1)
            
            top_k_scores, top_k_tokens = torch.topk(scores, top_k, dim=-1)
            
            degeneration_penalty = self._compute_degeneration_penalty(
                input_ids,
                top_k_tokens,
                past_key_values
            )
            
            contrastive_scores = (
                (1 - penalty_alpha) * top_k_scores -
                penalty_alpha * degeneration_penalty
            )
            
            selected_idx = contrastive_scores.argmax(-1)
            next_token = top_k_tokens.gather(-1, selected_idx.unsqueeze(-1))
            
            if next_token.item() == self.config.eos_token_id:
                break
            
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
        
        return prompt + self._decode(torch.tensor(generated_tokens))
    
    def _compute_degeneration_penalty(
        self,
        input_ids: torch.Tensor,
        candidate_tokens: torch.Tensor,
        past_key_values: Optional[List]
    ) -> torch.Tensor:
        """计算退化惩罚"""
        batch_size, seq_len = input_ids.shape
        num_candidates = candidate_tokens.shape[-1]
        
        penalty = torch.zeros(batch_size, num_candidates, device=self.device)
        
        for i in range(num_candidates):
            candidate = candidate_tokens[:, i]
            
            for j in range(seq_len):
                penalty[:, i] += (input_ids[:, j] == candidate).float()
        
        penalty = penalty / seq_len
        return penalty
