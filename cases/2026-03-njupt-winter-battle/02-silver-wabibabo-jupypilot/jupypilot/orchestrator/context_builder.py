from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..config import Config
from ..types import ChatMessage, SessionState
from .prompt_registry import PromptRegistry


def _count_cjk(s: str) -> int:
    n = 0
    for ch in s:
        o = ord(ch)
        # Basic CJK blocks (rough).
        if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            n += 1
    return n


@dataclass(frozen=True)
class TokenEstimator:
    """
    Heuristic token estimator (fallback when tokenizer is unavailable).
    - code/english: ~len/4
    - chinese: ~len/2 (more conservative)
    """

    def estimate(self, text: str) -> int:
        if not text:
            return 0
        cjk = _count_cjk(text)
        if cjk > max(20, len(text) // 6):
            return (len(text) + 1) // 2
        return (len(text) + 3) // 4


def _trim_to_token_budget(text: str, estimator: TokenEstimator, max_tokens: int) -> str:
    if estimator.estimate(text) <= max_tokens:
        return text
    # naive binary search on characters
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if estimator.estimate(text[:mid]) <= max_tokens:
            lo = mid + 1
        else:
            hi = mid
    cut = max(0, lo - 1)
    return text[:cut] + "\n...[TRIMMED]..."


@dataclass
class ContextBuilder:
    config: Config
    prompts: PromptRegistry
    estimator: TokenEstimator = TokenEstimator()

    def build(
        self,
        session: SessionState,
        *,
        task_kind: str,
        retrieved_context: str = "",
        extra_system: str = "",
    ) -> list[ChatMessage]:
        """
        Build messages for the current LLM call.
        The current user message should already be present in session.history.
        """
        ctx_cfg = self.config.context
        # 赛题③ 超长上下文：动态截断与 token 预算（max_ctx_tokens - completion_budget_tokens），优先保留 system+任务+最近轮次。
        input_budget = ctx_cfg.max_ctx_tokens - ctx_cfg.completion_budget_tokens

        system_parts: list[str] = []
        system_base = self.prompts.get_system_base()
        system_parts.append(system_base)

        # Global persona for beginner-friendly tutor (if configured).
        persona = self.prompts.get("persona_newbie_tutor")
        if persona:
            system_parts.append("[PERSONA]\n" + persona.strip())

        # Dynamic system hints.
        dyn: dict[str, Any] = {
            "repo_path": session.repo_path,
            "write_enabled": bool(session.flags.get("write_enabled", False)),
        }
        system_parts.append("[SYSTEM_META]\n" + json.dumps(dyn, ensure_ascii=False))

        if session.pinned_requirements:
            pins = "\n".join(f"- {x}" for x in session.pinned_requirements)
            system_parts.append("[PINNED_REQUIREMENTS]\n" + pins)

        ms = session.memory_summary
        if any([ms.constraints, ms.decisions, ms.progress, ms.todo, ms.pitfalls, ms.context.strip()]):
            system_parts.append(
                "[MEMORY_SUMMARY]\n"
                + json.dumps(
                    {
                        "constraints": ms.constraints,
                        "decisions": ms.decisions,
                        "progress": ms.progress,
                        "todo": ms.todo,
                        "pitfalls": ms.pitfalls,
                        "context": ms.context,
                    },
                    ensure_ascii=False,
                )
            )

        task_prompt = self.prompts.get_task_prompt(task_kind)
        if task_prompt:
            system_parts.append("[TASK]\n" + task_prompt.strip())

        if extra_system:
            system_parts.append("[EXTRA]\n" + extra_system.strip())

        if retrieved_context:
            system_parts.append("[RETRIEVED_CONTEXT]\n" + retrieved_context.strip())

        system_text = "\n\n".join(system_parts).strip()
        system_msg: ChatMessage = {"role": "system", "content": system_text}

        # 赛题③ 超长上下文-分段滚动：滑窗保留最近 history_max_turns*4 条，在 token 预算内裁剪。
        history = session.history[-max(1, ctx_cfg.history_max_turns * 4) :]

        system_tokens = self.estimator.estimate(system_msg["content"])
        if not history:
            if system_tokens > input_budget and retrieved_context:
                trimmed_rc = _trim_to_token_budget(
                    retrieved_context, self.estimator, max_tokens=max(256, input_budget // 2)
                )
                return self.build(session, task_kind=task_kind, retrieved_context=trimmed_rc, extra_system=extra_system)
            return [system_msg]

        # Always include the latest message (usually the current user request).
        latest = history[-1]
        latest_tokens = self.estimator.estimate(latest["content"])

        if system_tokens + latest_tokens > input_budget:
            if retrieved_context:
                # Trim retrieved context aggressively to make room for the latest user message.
                rc_tokens = self.estimator.estimate(retrieved_context)
                if rc_tokens <= 128:
                    return self.build(session, task_kind=task_kind, retrieved_context="", extra_system=extra_system)
                max_rc_tokens = max(64, input_budget - latest_tokens - max(512, system_tokens // 2))
                trimmed_rc = _trim_to_token_budget(retrieved_context, self.estimator, max_tokens=max_rc_tokens)
                # If trimming did not reduce, drop it to avoid infinite recursion.
                if trimmed_rc == retrieved_context:
                    return self.build(session, task_kind=task_kind, retrieved_context="", extra_system=extra_system)
                return self.build(session, task_kind=task_kind, retrieved_context=trimmed_rc, extra_system=extra_system)

            # As a last resort, trim the latest message to fit.
            max_latest_tokens = max(128, input_budget - system_tokens)
            trimmed_latest = dict(latest)
            trimmed_latest["content"] = _trim_to_token_budget(latest["content"], self.estimator, max_tokens=max_latest_tokens)
            history = history[:-1] + [trimmed_latest]
            latest = history[-1]
            latest_tokens = self.estimator.estimate(latest["content"])

        messages: list[ChatMessage] = [system_msg]
        total = system_tokens

        # Include as many recent messages as possible, but always keep latest.
        selected: list[ChatMessage] = [latest]
        total += latest_tokens
        for m in reversed(history[:-1]):
            t = self.estimator.estimate(m["content"])
            if total + t > input_budget:
                continue
            selected.append(m)
            total += t
        selected.reverse()
        messages.extend(selected)
        return messages
