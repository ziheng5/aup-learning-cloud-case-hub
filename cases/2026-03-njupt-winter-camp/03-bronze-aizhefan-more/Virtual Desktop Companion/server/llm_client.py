import json
import asyncio
import time
from typing import Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass

import httpx

from config import settings


@dataclass
class LLMResponse:
    text: str
    model: str
    duration_ms: float
    tokens_used: int
    raw_response: Dict[str, Any]
    parsed_data: Optional[Dict[str, Any]] = None


class OllamaClient:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL
        self.num_ctx = settings.OLLAMA_NUM_CTX
        self.temperature = settings.OLLAMA_TEMPERATURE
        self.timeout = httpx.Timeout(120.0, connect=10.0)
    
    async def generate(self, prompt: str, system_prompt: str = None,
                       stream: bool = False, context: list = None,
                       temperature: float = None) -> LLMResponse:
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": temperature if temperature is not None else self.temperature,
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        if context:
            payload["context"] = context
        
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
        
        duration_ms = (time.time() - start_time) * 1000
        
        text = data.get("response", "")
        tokens_used = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
        
        parsed_data = None
        try:
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                parsed_data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
        
        return LLMResponse(
            text=text,
            model=self.model,
            duration_ms=duration_ms,
            tokens_used=tokens_used,
            raw_response=data,
            parsed_data=parsed_data
        )
    
    async def chat(self, messages: list, stream: bool = False,
                   temperature: float = None) -> LLMResponse:
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": temperature if temperature is not None else self.temperature,
            }
        }
        
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
        
        duration_ms = (time.time() - start_time) * 1000
        
        message = data.get("message", {})
        text = message.get("content", "")
        tokens_used = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
        
        parsed_data = None
        try:
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                parsed_data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
        
        return LLMResponse(
            text=text,
            model=self.model,
            duration_ms=duration_ms,
            tokens_used=tokens_used,
            raw_response=data,
            parsed_data=parsed_data
        )
    
    async def generate_stream(self, prompt: str, system_prompt: str = None,
                              temperature: float = None) -> AsyncGenerator[str, None]:
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": temperature if temperature is not None else self.temperature,
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            token = data.get("response", "")
                            if token:
                                yield token
                        except json.JSONDecodeError:
                            pass


class DeepSeekClient:
    def __init__(self):
        self.api_key = settings.DEEPSEEK_API_KEY
        self.base_url = settings.DEEPSEEK_BASE_URL
        self.model = settings.DEEPSEEK_MODEL
        self.timeout = httpx.Timeout(120.0, connect=10.0)
    
    async def generate(self, prompt: str, system_prompt: str = None,
                       temperature: float = 0.7, max_tokens: int = 2000) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
        
        duration_ms = (time.time() - start_time) * 1000
        
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        text = message.get("content", "")
        
        usage = data.get("usage", {})
        tokens_used = usage.get("total_tokens", 0)
        
        parsed_data = None
        try:
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                parsed_data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
        
        return LLMResponse(
            text=text,
            model=self.model,
            duration_ms=duration_ms,
            tokens_used=tokens_used,
            raw_response=data,
            parsed_data=parsed_data
        )


class LLMClient:
    def __init__(self):
        self.ollama = OllamaClient()
        self.deepseek = DeepSeekClient()
        self.use_deepseek_for_complex = settings.USE_DEEPSEEK_FOR_COMPLEX
    
    def _should_use_deepseek(self, complexity: float, mode: str = "chat") -> bool:
        if mode == "wsl2":
            return True  # WSL2模式强制使用DeepSeek
        if not self.use_deepseek_for_complex:
            return False
        return complexity > 0.6
    
    async def generate(self, prompt: str, system_prompt: str = None,
                       complexity: float = 0.5, temperature: float = None,
                       stream: bool = False, mode: str = "chat") -> LLMResponse:
        
        if self._should_use_deepseek(complexity, mode):
            try:
                return await self.deepseek.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature if temperature else 0.7
                )
            except Exception as e:
                print(f"DeepSeek failed, falling back to Ollama: {e}")
        
        return await self.ollama.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            stream=stream,
            temperature=temperature
        )
    
    async def chat(self, messages: list, complexity: float = 0.5,
                   temperature: float = None, stream: bool = False) -> LLMResponse:
        
        if self._should_use_deepseek(complexity):
            try:
                prompt = messages[-1]["content"] if messages else ""
                system_prompt = messages[0]["content"] if len(messages) > 1 else None
                return await self.deepseek.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature if temperature else 0.7
                )
            except Exception as e:
                print(f"DeepSeek failed, falling back to Ollama: {e}")
        
        return await self.ollama.chat(
            messages=messages,
            stream=stream,
            temperature=temperature
        )
    
    async def generate_stream(self, prompt: str, system_prompt: str = None,
                              temperature: float = None) -> AsyncGenerator[str, None]:
        async for token in self.ollama.generate_stream(prompt, system_prompt, temperature):
            yield token
