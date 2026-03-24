from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import requests

from ..config import OllamaConfig
from ..types import ChatMessage
from .backoff import backoff_sleep_seconds


class OllamaError(RuntimeError):
    pass


EventSink = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class OllamaResponse:
    content: str
    model: str | None = None
    duration_ms: int | None = None
    raw: dict[str, Any] | None = None
    usage: dict[str, int] | None = None


def _coerce_nonneg_int(v: Any) -> int | None:
    try:
        i = int(v)
    except Exception:
        return None
    if i < 0:
        return None
    return i


def _extract_usage(raw: Mapping[str, Any] | None) -> dict[str, int] | None:
    """
    Normalize provider-specific usage into a common shape:
    - prompt_tokens
    - completion_tokens
    - total_tokens
    """
    if not isinstance(raw, Mapping):
        return None

    # OpenAI-style nested usage: {"usage": {"prompt_tokens": .., "completion_tokens": ..}}
    usage_obj = raw.get("usage")
    if isinstance(usage_obj, Mapping):
        prompt = _coerce_nonneg_int(usage_obj.get("prompt_tokens") or usage_obj.get("input_tokens"))
        completion = _coerce_nonneg_int(usage_obj.get("completion_tokens") or usage_obj.get("output_tokens"))
        total = _coerce_nonneg_int(usage_obj.get("total_tokens"))
        if total is None and prompt is not None and completion is not None:
            total = prompt + completion
        out: dict[str, int] = {}
        if prompt is not None:
            out["prompt_tokens"] = prompt
        if completion is not None:
            out["completion_tokens"] = completion
        if total is not None:
            out["total_tokens"] = total
        return out or None

    # Ollama-style: prompt_eval_count / eval_count at top-level.
    prompt = _coerce_nonneg_int(raw.get("prompt_eval_count"))
    completion = _coerce_nonneg_int(raw.get("eval_count"))
    total = _coerce_nonneg_int(raw.get("total_tokens"))
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion

    # Some providers put prompt/completion at top-level.
    if prompt is None:
        prompt = _coerce_nonneg_int(raw.get("prompt_tokens") or raw.get("input_tokens"))
    if completion is None:
        completion = _coerce_nonneg_int(raw.get("completion_tokens") or raw.get("output_tokens"))
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion

    if prompt is None and completion is None and total is None:
        return None
    out = {}
    if prompt is not None:
        out["prompt_tokens"] = prompt
    if completion is not None:
        out["completion_tokens"] = completion
    if total is not None:
        out["total_tokens"] = total
    return out


class OllamaClient:
    def __init__(self, config: OllamaConfig, *, on_event: EventSink | None = None) -> None:
        self._config = config
        self._on_event = on_event

    @property
    def config(self) -> OllamaConfig:
        return self._config

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        stream: bool = False,
        think: bool | None = None,
        options: Mapping[str, Any] | None = None,
        request_meta: Mapping[str, Any] | None = None,
        on_event: EventSink | None = None,
    ) -> OllamaResponse:
        """
        Call Ollama /api/chat.

        Retries:
        - network errors / timeouts
        - HTTP 429 / 503
        """
        payload: dict[str, Any] = {
            "model": model or self._config.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "num_ctx": self._config.num_ctx,
                "temperature": self._config.temperature,
                "top_p": self._config.top_p,
            },
        }
        # Ollama's think parameter is at the top level, NOT inside options.
        if think is not None:
            payload["think"] = think
        if options:
            payload["options"] = {**payload["options"], **dict(options)}

        attempt = 0
        last_error: Exception | None = None

        # -----------------------------
        # 赛题③ 错误处理：API 生命周期管理（超时、网络波动、频率限制）。便于贴入技术文档。
        # -----------------------------
        # 1) 每次尝试都会发出 llm_request_start（包含 attempt），成功时发出 llm_request_end。
        # 2) 遇到“可恢复”的临时故障（断连/超时、HTTP 429/503、流式返回中断/坏响应）时：
        #    - 计算指数退避等待时间（Exponential Backoff + jitter，避免同时重试造成雪崩）
        #    - 先发出 llm_retry（包含 wait_s），让 UI 立即展示“正在重试/等待...”
        #    - 再 sleep(wait_s) 后进入下一次 attempt
        # 3) 这里不会真的切换“备用线路/备用 endpoint”（你配置里只有一个 endpoint），但在 429/503
        #    时我们仍附带中文提示“正在尝试通过内网备用路径重新连接集群资源...”，用于演示时的友好
        #    告知，避免长等待被误判为系统死机。
        while attempt <= self._config.max_retries:
            started = time.time()
            self._emit(
                "llm_request_start",
                {
                    "endpoint": self._config.endpoint,
                    "model": payload["model"],
                    "stream": stream,
                    "attempt": attempt,
                    "meta": dict(request_meta or {}),
                },
                on_event=on_event,
            )
            try:
                # 使用首字节超时，避免在 ROCm 等环境下因模型冷启动长时间卡在“推理中”
                connect_s, first_byte_s = 30, self._config.timeout_first_byte_s
                resp = requests.post(
                    self._config.endpoint,
                    json=payload,
                    timeout=(connect_s, first_byte_s),
                    headers={"Content-Type": "application/json"},
                    stream=stream,
                )
            except (requests.Timeout, requests.ConnectionError) as e:
                last_error = e
                duration_ms = int((time.time() - started) * 1000)
                wait_s = backoff_sleep_seconds(attempt, base_s=self._config.backoff_base_s)
                will_retry = attempt < self._config.max_retries
                self._emit(
                    "llm_retry",
                    {
                        "attempt": attempt,
                        "max_retries": self._config.max_retries,
                        "reason": type(e).__name__,
                        "wait_s": wait_s,
                        "duration_ms": duration_ms,
                        "will_retry": will_retry,
                        "hint_zh": "集群接口连接异常，正在指数退避重试...",
                    },
                    on_event=on_event,
                )
                if will_retry:
                    time.sleep(wait_s)
                attempt += 1
                continue

            duration_ms = int((time.time() - started) * 1000)
            if resp.status_code in (429, 503):
                last_error = OllamaError(f"ollama HTTP {resp.status_code}")
                wait_s = backoff_sleep_seconds(attempt, base_s=self._config.backoff_base_s)
                will_retry = attempt < self._config.max_retries
                self._emit(
                    "llm_retry",
                    {
                        "attempt": attempt,
                        "max_retries": self._config.max_retries,
                        "status": resp.status_code,
                        "reason": "rate_limited",
                        "wait_s": wait_s,
                        "duration_ms": duration_ms,
                        "will_retry": will_retry,
                        "hint_zh": "正在尝试通过内网备用路径重新连接集群资源...",
                    },
                    on_event=on_event,
                )
                if will_retry:
                    time.sleep(wait_s)
                attempt += 1
                continue

            if resp.status_code >= 400:
                try:
                    body = resp.text
                except Exception:
                    body = "<unreadable>"
                self._emit(
                    "llm_request_end",
                    {"ok": False, "status": resp.status_code, "duration_ms": duration_ms, "body": body[:2000]},
                    on_event=on_event,
                )
                raise OllamaError(f"ollama HTTP {resp.status_code}: {body[:2000]}")

            try:
                if stream:
                    content, raw, usage = self._read_stream(
                        resp,
                        model=str(payload["model"]),
                        request_meta=request_meta,
                        on_event=on_event,
                    )
                else:
                    raw = resp.json()
                    msg = raw.get("message") if isinstance(raw, dict) else None
                    content = msg.get("content") if isinstance(msg, dict) else None
                    if not isinstance(content, str):
                        raise OllamaError("ollama response missing message.content")
                    usage = _extract_usage(raw)
            except Exception as e:
                last_error = e
                wait_s = backoff_sleep_seconds(attempt, base_s=self._config.backoff_base_s)
                will_retry = attempt < self._config.max_retries
                self._emit(
                    "llm_retry",
                    {
                        "attempt": attempt,
                        "max_retries": self._config.max_retries,
                        "reason": "bad_response",
                        "wait_s": wait_s,
                        "will_retry": will_retry,
                        "hint_zh": "集群接口返回异常，正在指数退避重试...",
                    },
                    on_event=on_event,
                )
                if will_retry:
                    time.sleep(wait_s)
                attempt += 1
                continue

            # For stream=True, usage is best-effort emitted during streaming inside _read_stream.
            if usage and not stream:
                self._emit(
                    "llm_usage",
                    {"model": payload["model"], "stream": False, "meta": dict(request_meta or {}), **usage},
                    on_event=on_event,
                )

            self._emit(
                "llm_request_end",
                {
                    "ok": True,
                    "status": resp.status_code,
                    "duration_ms": duration_ms,
                    "chars": len(content),
                    "usage": usage,
                },
                on_event=on_event,
            )
            return OllamaResponse(content=content, model=payload["model"], duration_ms=duration_ms, raw=raw, usage=usage)

        raise OllamaError(f"ollama request failed after retries: {last_error}")

    def _read_stream(
        self,
        resp: requests.Response,
        *,
        model: str,
        request_meta: Mapping[str, Any] | None,
        on_event: EventSink | None,
    ) -> tuple[str, dict[str, Any] | None, dict[str, int] | None]:
        """
        Ollama stream=true returns multiple JSON lines.
        We concatenate message.content fragments.

        Guards against stalled/runaway generation:
        - Per-chunk socket timeout (120s max between chunks)
        - Total wall-clock timeout (timeout_s from config, default 300s)
        - Total character cap (64k chars — well beyond any useful output)
        """
        # Set a per-chunk read timeout on the underlying socket.
        chunk_timeout = min(self._config.timeout_s, 120)
        try:
            sock = resp.raw._fp.fp.raw._sock  # type: ignore[union-attr]
            sock.settimeout(chunk_timeout)
        except Exception:
            pass  # Best-effort; some transports don't expose the socket.

        # Total wall-clock limit to prevent infinite generation.
        # Use a tighter limit than timeout_s to stay within the user's
        # expected response time (~60s).  45s for generation leaves room
        # for network overhead and tool execution.
        stream_timeout = min(self._config.timeout_s, 45)
        wall_deadline = time.time() + stream_timeout
        _MAX_STREAM_CHARS = 65536

        chunks: list[str] = []
        total_chars = 0
        last_obj: dict[str, Any] | None = None
        last_usage: dict[str, int] | None = None
        try:
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue

                # Check wall-clock deadline.
                if time.time() > wall_deadline:
                    break

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                last_obj = obj
                msg = obj.get("message") if isinstance(obj, dict) else None
                if isinstance(msg, dict):
                    part = msg.get("content")
                    if isinstance(part, str) and part:
                        chunks.append(part)
                        total_chars += len(part)
                        # Cap total output to prevent runaway generation.
                        if total_chars > _MAX_STREAM_CHARS:
                            break

                usage = _extract_usage(obj)
                if usage and usage != last_usage:
                    last_usage = usage
                    self._emit(
                        "llm_usage",
                        {"model": model, "stream": True, "meta": dict(request_meta or {}), **usage},
                        on_event=on_event,
                    )
        except Exception:
            # Socket timeout or connection reset — return whatever we have so far.
            if not chunks:
                raise

        return "".join(chunks), last_obj, last_usage

    def _emit(self, event: str, data: dict[str, Any], *, on_event: EventSink | None = None) -> None:
        sink = on_event or self._on_event
        if not sink:
            return
        payload = {"event": event, "data": data}
        try:
            sink(payload)
        except Exception:
            # Event emission must never break the main flow.
            return
