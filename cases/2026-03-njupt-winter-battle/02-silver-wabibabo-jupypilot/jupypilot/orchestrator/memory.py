from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from ..config import Config
from ..llm.ollama_client import OllamaClient
from ..types import ChatMessage, SessionState
from .prompt_registry import PromptRegistry
from .validator import EnvelopeParseError, ValidationError, parse_envelope, validate_memory_summary_schema
from .context_builder import TokenEstimator


class MemorySummaryError(RuntimeError):
    pass


EventSink = Callable[[str, dict[str, Any]], None]


def _format_history_slice(messages: list[ChatMessage]) -> str:
    parts: list[str] = []
    for m in messages:
        parts.append(f"[{m['role']}] {m['content']}")
    return "\n\n".join(parts)


@dataclass
class MemoryManager:
    config: Config
    llm: OllamaClient
    prompts: PromptRegistry
    estimator: TokenEstimator = TokenEstimator()

    def maybe_summarize(self, session: SessionState, *, emit: EventSink | None = None) -> None:
        """
        If history exceeds token threshold, summarize older messages into session.memory_summary.
        Pinned requirements are not summarized away; they are injected separately by ContextBuilder.
        赛题③ 错误处理-超长上下文：级联摘要（Recursive Summarization），超 memory_summary_trigger_tokens 时压缩旧历史为结构化 JSON。
        """
        threshold = self.config.context.memory_summary_trigger_tokens
        if not session.history:
            return
        total = sum(self.estimator.estimate(m["content"]) for m in session.history)
        if total < threshold:
            return

        # Keep the most recent messages; summarize the earliest slice.
        keep = max(8, self.config.context.history_max_turns * 2)
        if len(session.history) <= keep:
            return
        slice_msgs = session.history[: -keep]
        if not slice_msgs:
            return

        if emit:
            emit("memory_summary_start", {"messages": len(slice_msgs), "estimated_tokens": total})

        system = self.prompts.get_system_base() + "\n\n[TASK]\n" + self.prompts.get_task_prompt("memory_summary")
        # Extra hard rule: final only.
        system += "\n\n[HARD_RULE]\n你只能输出 kind=final 的单行 JSON，不得输出任何 tool 调用。"

        user = "[需要摘要的历史]\n" + _format_history_slice(slice_msgs)
        messages: list[ChatMessage] = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        correction_left = self.config.tool_loop.correction_retries
        last_err: str | None = None
        for attempt in range(correction_left + 1):
            resp = self.llm.chat(
                messages,
                stream=True,
                think=False,
                request_meta={"task": "memory_summary", "attempt": attempt, "session_id": session.session_id},
                on_event=(lambda p: emit(p["event"], p["data"])) if emit else None,
            )
            try:
                env = parse_envelope(resp.content)
                if env["kind"] != "final" or env["format"] != "json":
                    raise ValidationError("历史摘要输出要求 final.format=json")
                obj = json.loads(env["content"])
                validate_memory_summary_schema(obj)
            except (EnvelopeParseError, ValidationError, json.JSONDecodeError) as e:
                last_err = str(e)
                if attempt >= correction_left:
                    break
                # ask for correction
                messages.append(
                    {
                        "role": "system",
                        "content": f"你上一次输出不符合协议：{e}。请严格输出【单行 JSON】final.format=json，不要包含任何其它字符。",
                    }
                )
                continue

            self._merge_into_session(session, obj)
            # Drop summarized slice from history.
            session.history = session.history[-keep:]
            if emit:
                emit("memory_summary_end", {"ok": True, "kept_messages": keep})
            return

        if emit:
            emit("memory_summary_end", {"ok": False, "error": last_err or "unknown"})

    def _merge_into_session(self, session: SessionState, obj: dict[str, Any]) -> None:
        ms = session.memory_summary
        ms.constraints = _merge_unique(ms.constraints, obj.get("constraints", []))
        ms.decisions = _merge_unique(ms.decisions, obj.get("decisions", []))
        ms.progress = _merge_unique(ms.progress, obj.get("progress", []))
        ms.todo = _merge_unique(ms.todo, obj.get("todo", []))
        ms.pitfalls = _merge_unique(ms.pitfalls, obj.get("pitfalls", []))


def _merge_unique(existing: list[str], incoming: Any) -> list[str]:
    # Use set for O(1) membership check; otherwise repeated "x not in out" is O(n) per item → O(n²).
    seen = set(existing)
    out = list(existing)
    if isinstance(incoming, list):
        for x in incoming:
            if isinstance(x, str) and x not in seen:
                seen.add(x)
                out.append(x)
    return out
