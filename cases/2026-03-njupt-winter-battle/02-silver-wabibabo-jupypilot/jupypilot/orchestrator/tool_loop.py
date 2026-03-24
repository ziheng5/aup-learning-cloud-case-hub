from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from ..config import Config
from ..llm.ollama_client import OllamaClient
from ..tools.runtime import ToolRuntime
from ..types import Envelope, FinalResponse, SessionState
from .context_builder import ContextBuilder
from .memory import MemoryManager
from .policy_guard import PolicyGuard, PolicyViolation
from .validator import (
    EnvelopeParseError,
    ValidationError,
    _try_fix_diff_header,
    _try_fix_json_invalid_escapes,
    extract_single_diff_block,
    parse_envelope,
    parse_envelopes,
    parse_json_content,
    validate_diff_contract,
    validate_refactor_suggestions_schema,
    validate_scaffold_plan_schema,
)


EventSink = Callable[[str, dict[str, Any]], None]


class MaxToolItersError(RuntimeError):
    pass


_EVIDENCE_RE = re.compile(r"\b[^\s:]+:\d+\b")


def _require_mentor_section_in_markdown(content: str, task_kind: str) -> None:
    """要求 markdown content 必须包含「导师详细讲解」部分，否则视为校验失败并触发重试。"""
    if "导师详细讲解" in content or "Mentor's Deep Dive" in content:
        return
    raise ValidationError(
        f"{task_kind} 输出必须包含「导师详细讲解」部分（标题 ## 👨‍🏫 导师详细讲解 或含 Mentor's Deep Dive），不可只输出技术分析即结束。请补全第二部分后重新输出。"
    )


def _require_mentor_deep_dive_in_json(obj: dict[str, Any], task_kind: str) -> None:
    """要求 JSON 对象必须包含非空的 mentor_deep_dive 字段。"""
    val = obj.get("mentor_deep_dive")
    if isinstance(val, str) and val.strip():
        return
    raise ValidationError(
        f"{task_kind} 输出必须包含 mentor_deep_dive 字段（导师详细讲解），且为非空字符串。"
    )


# Simplified task prompts for single-turn direct mode.
# These are intentionally short and simple — no JSON envelope, no tool
# instructions.  qwen3-coder handles these much better than the full
# multi-turn protocol prompts.
_SIMPLE_TASK_PROMPTS: dict[str, str] = {
    "code_qa": (
        "你是代码分析助手。请用中文回答用户关于代码的问题，要求："
        "（1）尽量详细、分点说明各配置类/函数的作用与参数；"
        "（2）涉及配置加载、示例用法时请一并写出；"
        "（3）引用具体的文件路径和行号作为证据。"
    ),
    "code_patch": (
        "你是代码修复助手。请分析代码中的问题，输出修复方案。\n"
        "用 ```diff 代码块输出 unified diff 格式的补丁，包含完整的 diff --git 头。"
    ),
    "testgen": "你是测试生成助手。请为以下代码生成 pytest 测试用例，用 markdown 代码块输出。",
    "refactor": "你是代码审查助手。请分析以下代码的问题，给出重构建议，说明文件路径、行号、问题和建议。",
    "refactor_diff": (
        "你是代码重构助手。请输出重构补丁。\n"
        "用 ```diff 代码块输出 unified diff 格式，包含完整的 diff --git 头。"
    ),
}


@dataclass
class ToolLoopEngine:
    config: Config
    llm: OllamaClient
    tools: ToolRuntime
    policy: PolicyGuard
    context_builder: ContextBuilder
    memory: MemoryManager

    def run(
        self,
        session: SessionState,
        *,
        task_kind: str,
        user_text: str,
        retrieved_context: str = "",
        emit: EventSink | None = None,
    ) -> FinalResponse:
        if task_kind in {"code_qa", "code_patch", "testgen", "refactor", "refactor_diff", "scaffold"}:
            self.policy.check_user_input(user_text)

        # ── Direct mode ──────────────────────────────────────────────
        # For tasks that already have RAG-retrieved code, skip the tool
        # loop entirely and do a single-turn call.  This avoids the
        # multi-turn failure mode where qwen3-coder chokes after tool
        # calls.  The tool loop is only used as a fallback if direct
        # mode fails (e.g. RAG didn't find the right code).
        # NOTE: code_qa 强制走 Tool-loop，多轮使用工具获取证据（search_code/open_file），
        # Direct 单轮仅作为 patch/testgen/refactor/refactor_diff 的性能优化与兜底路径。
        direct_mode_tasks = {"code_patch", "testgen", "refactor", "refactor_diff"}
        if retrieved_context and task_kind in direct_mode_tasks:
            if emit:
                emit("direct_mode_start", {"task": task_kind, "rag_chars": len(retrieved_context)})
            pseudo_results = [{
                "ok": True, "tool": "open_file",
                "data": {"path": "(RAG)", "content": retrieved_context},
                "error": None,
            }]
            result = self._single_turn_fallback(
                session, task_kind, user_text, pseudo_results, emit,
            )
            if result is not None:
                if emit:
                    emit("direct_mode_ok", {"task": task_kind, "chars": len(result.get("content", ""))})
                # Still add user_text to history for session continuity.
                session.history.append({"role": "user", "content": user_text})
                session.history.append({"role": "assistant", "content": result.get("content", "")[:1500]})
                return result
            if emit:
                emit("direct_mode_failed", {"task": task_kind, "reason": "empty_or_error"})
            # Fall through to tool loop.

        session.history.append({"role": "user", "content": user_text})
        correction_left = self.config.tool_loop.correction_retries
        correction_note = ""
        # Track tool results for graceful fallback when model chokes.
        last_tool_results: list[dict[str, Any]] = []

        max_iters = self.config.tool_loop.max_iters
        for i in range(max_iters):
            session.stats.tool_iters += 1
            if emit:
                emit("tool_loop_iter", {"iter": i, "task": task_kind})

            # Summarize if needed.
            self.memory.maybe_summarize(session, emit=(lambda e, d: emit(e, d)) if emit else None)

            messages = self.context_builder.build(
                session,
                task_kind=task_kind,
                retrieved_context=retrieved_context,
                extra_system=correction_note,
            )

            # DEBUG: inspect the exact messages sent to LLM each turn.
            print(
                "\n========== [DEBUG] 发送给 LLM 的 Messages ==========\n",
                messages,
                "\n==================================================\n",
                flush=True,
            )

            session.stats.llm_calls += 1
            # Disable thinking mode for all tasks to reduce GPU load.
            # qwen3-coder generates huge <think> blocks that waste time and
            # produce no useful output for structured JSON tool-call workflows.
            resp = self.llm.chat(
                messages,
                stream=True,
                think=False,
                request_meta={"task": task_kind, "iter": i, "session_id": session.session_id},
                on_event=(lambda p: emit(p["event"], p["data"])) if emit else None,
            )
            raw = resp.content.strip()

            if emit:
                emit("llm_response", {"chars": len(raw), "raw_content": raw[:2000]})

            # Guard: if model returned empty content, treat as invalid and retry.
            if not raw:
                # Strategy 1 (primary): If we have tool results, bypass the
                # tool loop entirely and make a fresh single-turn LLM call
                # with code content embedded directly in the prompt.
                # This avoids the multi-turn problem where qwen3-coder returns
                # empty after seeing tool results in conversation history.
                if last_tool_results:
                    if emit:
                        emit("single_turn_fallback_start", {"reason": "empty_after_tool_results", "tool_results": len(last_tool_results)})
                    result = self._single_turn_fallback(
                        session, task_kind, user_text, last_tool_results, emit,
                    )
                    if result is not None:
                        return result
                    # single-turn also failed → fall through to _build_fallback_response
                    if emit:
                        emit("fallback_triggered", {"reason": "single_turn_also_failed", "tool_results": len(last_tool_results)})
                    return _build_fallback_response(last_tool_results, reason="empty")

                # Strategy 2: No tool results yet — normal correction retry.
                if correction_left > 0:
                    correction_left -= 1
                    correction_note = "你上一次输出为空。请严格输出【单行 JSON】tool 或 final，不要包含任何其它字符。"
                    if emit:
                        emit("llm_output_invalid", {"reason": "empty_response", "correction_left": correction_left, "raw_content": resp.content[:2000]})
                    continue

                # Strategy 3: No tool results, retries exhausted — try a
                # single-turn call using retrieved_context as the code content.
                # This covers the case where the model chokes on the very first
                # turn (before any tool calls) due to context/prompt issues.
                if retrieved_context:
                    if emit:
                        emit("single_turn_fallback_start", {"reason": "empty_no_tools_rag_fallback", "rag_chars": len(retrieved_context)})
                    pseudo_results = [{
                        "ok": True, "tool": "open_file",
                        "data": {"path": "(RAG retrieved)", "content": retrieved_context},
                        "error": None,
                    }]
                    result = self._single_turn_fallback(
                        session, task_kind, user_text, pseudo_results, emit,
                    )
                    if result is not None:
                        return result

                # All strategies exhausted — graceful fallback.
                if emit:
                    emit("fallback_triggered", {"reason": "empty_response_exhausted", "tool_results": len(last_tool_results)})
                return _build_fallback_response(last_tool_results, reason="empty")

            try:
                envelopes = parse_envelopes(raw)
            except EnvelopeParseError as e:
                if correction_left > 0:
                    correction_left -= 1
                    correction_note = f"你上一次输出不符合协议：{e}。请严格输出【单行 JSON】tool 或 final，不要包含任何其它字符。"
                    if emit:
                        emit("llm_output_invalid", {"reason": str(e), "correction_left": correction_left, "raw_content": raw[:2000]})
                    continue
                # Retries exhausted — try single-turn fallback if we have tool results.
                if last_tool_results:
                    if emit:
                        emit("single_turn_fallback_start", {"reason": "parse_error_exhausted", "tool_results": len(last_tool_results)})
                    result = self._single_turn_fallback(
                        session, task_kind, user_text, last_tool_results, emit,
                    )
                    if result is not None:
                        return result
                if emit:
                    emit("fallback_triggered", {"reason": "parse_error_exhausted", "tool_results": len(last_tool_results)})
                return _build_fallback_response(last_tool_results, reason="parse_error")

            # Store the model output for replay.
            # 若本轮有多个 tool envelope，全部序列化到一条 assistant 消息（换行分隔），
            # 以便与后续多条 user（工具结果）一一对应；若仅有 final 则存该 final；否则存 raw。
            # Single pass: collect tool_envs and first final to avoid iterating envelopes twice.
            tool_envs: list[Envelope] = []
            final_env: FinalResponse | None = None
            for e in envelopes:
                if e.get("kind") == "tool":
                    tool_envs.append(e)
                elif e.get("kind") == "final" and final_env is None:
                    final_env = e  # type: ignore[assignment]
            if tool_envs:
                history_content = "\n".join(json.dumps(e, ensure_ascii=False) for e in tool_envs)
                session.history.append({"role": "assistant", "content": history_content})
            elif final_env is not None:
                session.history.append({"role": "assistant", "content": json.dumps(final_env, ensure_ascii=False)})
            else:
                session.history.append({"role": "assistant", "content": raw})

            # Process all envelopes: execute tool calls, then handle final.
            # 赛题③ 工具调用幻觉防护（半严格拦截法）：
            # 如果同一轮中既出现 tool 又出现 final，则视为协议违规：
            # - 仍然执行本轮所有 tool 调用，确保工具副作用生效；
            # - 强制丢弃本轮所有 final，要求模型在下一轮基于最新工具结果重新作答。
            has_tool = bool(tool_envs)
            protocol_violation_tool_and_final = False

            final = None
            for env in envelopes:
                if env["kind"] == "tool":
                    call = env
                    if emit:
                        emit("tool_call", {"tool": call["tool"], "args": _safe_args_preview(call.get("args", {}))})
                    try:
                        self.policy.check_tool_call(call, session)
                    except PolicyViolation as e:
                        tool_result = {
                            "ok": False,
                            "tool": call["tool"],
                            "data": None,
                            "error": {"code": e.code, "message": str(e), "details": e.details},
                        }
                    else:
                        session.stats.tool_calls += 1
                        tool_result = self.tools.execute(call["tool"], call.get("args", {}) or {}, session)

                    last_tool_results.append(tool_result)
                    session.history.append(
                        {
                            "role": "user",
                            "content": "这是工具执行的返回结果，请根据此结果继续回答：\n"
                            + _truncate_tool_result_for_history(tool_result),
                        }
                    )
                    if emit:
                        emit(
                            "tool_result",
                            {
                                "tool": call["tool"],
                                "ok": tool_result.get("ok"),
                                "error": tool_result.get("error"),
                                "data_preview": _tool_result_preview(tool_result),
                            },
                        )
                elif env["kind"] == "final":
                    if has_tool:
                        # 协议违规：同一轮中同时输出 tool 和 final。
                        # 丢弃本轮 final，给出严厉的系统更正提示，要求模型在下一轮基于工具结果重新作答。
                        protocol_violation_tool_and_final = True
                        correction_note = (
                            "你本轮同时输出了工具调用与最终答案；系统已执行完所有工具并丢弃了本轮的 final。"
                            "请根据下方最新的工具执行结果，在下一轮回复中重新思考并只输出最终答案（kind=final）。"
                        )
                        if emit:
                            emit(
                                "llm_output_invalid",
                                {
                                    "reason": "tool_and_final_same_turn",
                                    "raw_content": raw[:2000],
                                },
                            )
                        # 不设置 final，继续处理剩余 envelopes（以便执行所有工具调用）。
                        continue

                    final = env
                    break  # Use the first final, ignore anything after it.

            # If no final was found, all envelopes were tool calls → next iteration.
            if final is None:
                # 仅在不存在「tool+final 同轮违规」的情况下清空 correction_note。
                if not protocol_violation_tool_and_final:
                    correction_note = ""
                continue

            try:
                self._validate_final(task_kind, final, user_text=user_text)

                # For diff-producing tasks, optionally run git apply --check and iterate if it fails.
                if task_kind in {"code_patch", "refactor_diff"} and self.config.tools.allow_git_apply_check:
                    diff = extract_single_diff_block(final["content"])
                    check = self.tools.execute("git_apply_check", {"diff": diff}, session)
                    session.stats.tool_calls += 1
                    session.history.append(
                        {
                            "role": "user",
                            "content": "这是工具执行的返回结果，请根据此结果继续回答：\n"
                            + json.dumps(check, ensure_ascii=False),
                        }
                    )
                    if emit:
                        emit(
                            "tool_result",
                            {
                                "tool": "git_apply_check",
                                "ok": check.get("ok"),
                                "error": check.get("error"),
                                "data_preview": _tool_result_preview(check),
                            },
                        )
                    data = check.get("data") or {}
                    if check.get("ok") and not bool(data.get("ok_to_apply", True)):
                        correction_note = "git apply --check 失败。请基于错误信息修正 diff，只输出新的 diff。"
                        continue

            except ValidationError as e:
                if correction_left > 0:
                    correction_left -= 1
                    correction_note = f"你上一次最终输出不合规：{e}。请严格按任务要求重发（仅单行 JSON envelope）。"
                    if emit:
                        emit("llm_output_invalid", {"reason": str(e), "correction_left": correction_left})
                    continue
                # Retries exhausted — try single-turn fallback if we have tool results.
                if last_tool_results:
                    if emit:
                        emit("single_turn_fallback_start", {"reason": "validation_exhausted", "tool_results": len(last_tool_results)})
                    result = self._single_turn_fallback(
                        session, task_kind, user_text, last_tool_results, emit,
                    )
                    if result is not None:
                        return result
                if emit:
                    emit("fallback_triggered", {"reason": "validation_exhausted", "tool_results": len(last_tool_results)})
                return _build_fallback_response(last_tool_results, reason="validation_error")

            if emit:
                emit("final_ok", {"task": task_kind, "format": final["format"], "chars": len(final["content"])})
            return final

        # Max iterations exhausted — force one last JSON-final-only call with full history.
        # 此时已经达到工具调用上限：不要丢弃多轮记忆，也不要走单轮无上下文兜底。
        # 追加一条用户消息，明确告知模型必须停止调用工具，只能基于现有上下文输出最终结论。
        forced_user_msg = (
            "系统强制中止：工具调用已达到最大轮次限制。请立刻停止调用任何工具，"
            "严格基于你目前已经收集到的上下文代码，总结你的发现并输出最终的结论"
            "（必须使用 {'kind':'final',...} JSON 格式）。如果信息不全，请诚实指出缺失的部分，绝不允许凭空捏造代码行号。"
        )
        session.history.append({"role": "user", "content": forced_user_msg})

        extra_system_final = (
            "注意：你已经达到本次会话允许的工具调用上限。接下来的一次回复中，"
            "严禁再调用任何工具（kind='tool'），只能输出一个 kind='final' 的 JSON envelope。"
        )

        if emit:
            emit(
                "max_iters_force_final_start",
                {"tool_results": len(last_tool_results)},
            )

        messages = self.context_builder.build(
            session,
            task_kind=task_kind,
            retrieved_context=retrieved_context,
            extra_system=extra_system_final,
        )

        print(
            "\n========== [DEBUG] 发送给 LLM 的 Messages (max_iters_force_final) ==========\n",
            messages,
            "\n==================================================\n",
            flush=True,
        )

        session.stats.llm_calls += 1
        try:
            resp = self.llm.chat(
                messages,
                stream=True,
                think=False,
                request_meta={"task": task_kind, "iter": "max_iters_force_final", "session_id": session.session_id},
                on_event=(lambda p: emit(p["event"], p["data"])) if emit else None,
            )
        except Exception as exc:
            if emit:
                emit("max_iters_force_final_error", {"error": str(exc)})
            # 退回到已有工具结果的友好兜底。
            return _build_fallback_response(last_tool_results, reason="max_iters")

        raw = resp.content.strip()
        if emit:
            emit("llm_response", {"chars": len(raw), "raw_content": raw[:2000]})

        if not raw:
            if emit:
                emit("fallback_triggered", {"reason": "max_iters_force_final_empty", "tool_results": len(last_tool_results)})
            return _build_fallback_response(last_tool_results, reason="max_iters")

        try:
            envelopes = parse_envelopes(raw)
        except EnvelopeParseError:
            if emit:
                emit("fallback_triggered", {"reason": "max_iters_force_final_parse_error", "tool_results": len(last_tool_results)})
            return _build_fallback_response(last_tool_results, reason="max_iters")

        # 只接受 final，忽略任何 tool（此时已禁止再调用工具）。
        final: FinalResponse | None = None
        for env in envelopes:
            if env["kind"] == "final":
                final = env
                break

        if final is None:
            if emit:
                emit("fallback_triggered", {"reason": "max_iters_force_final_no_final", "tool_results": len(last_tool_results)})
            return _build_fallback_response(last_tool_results, reason="max_iters")

        try:
            self._validate_final(task_kind, final, user_text=user_text)
        except ValidationError:
            if emit:
                emit("fallback_triggered", {"reason": "max_iters_force_final_validation_error", "tool_results": len(last_tool_results)})
            return _build_fallback_response(last_tool_results, reason="max_iters")

        if emit:
            emit("final_ok", {"task": task_kind, "format": final["format"], "chars": len(final["content"])})
        return final

    def _validate_final(self, task_kind: str, final: FinalResponse, *, user_text: str = "") -> None:
        if task_kind in {"code_patch", "refactor_diff"}:
            if final["format"] != "markdown":
                raise ValidationError("补丁类任务要求 final.format=markdown")
            if task_kind == "code_patch":
                _require_mentor_section_in_markdown(final["content"], "code_patch")
            diff = extract_single_diff_block(final["content"])
            # Auto-fix common diff format issues (missing git header, etc.)
            # before validation to avoid costly retry rounds.
            fixed = _try_fix_diff_header(diff)
            if fixed != diff:
                # Replace the diff block in the final content with the fixed version.
                final["content"] = final["content"].replace(diff, fixed)
                diff = fixed
            validate_diff_contract(diff)
            return

        if task_kind == "testgen":
            if final["format"] != "markdown":
                raise ValidationError("testgen 要求 final.format=markdown")
            # 强制要求 content 包含「导师详细讲解」部分，保证稳定输出两段式。
            _require_mentor_section_in_markdown(final["content"], "testgen")
            return

        if task_kind == "code_qa":
            if final["format"] != "markdown":
                raise ValidationError("code_qa 要求 final.format=markdown")
            if len(user_text) > 10 and not _EVIDENCE_RE.search(final["content"]):
                raise ValidationError("code_qa 输出必须包含至少 1 条证据引用（path:line）")
            _require_mentor_section_in_markdown(final["content"], "code_qa")
            return

        if task_kind == "scaffold":
            if final["format"] != "json":
                raise ValidationError("scaffold 要求 final.format=json")
            try:
                obj = json.loads(final["content"])
            except json.JSONDecodeError as e:
                fixed = _try_fix_json_invalid_escapes(final["content"])
                if fixed == final["content"]:
                    raise ValidationError(f"final.content 必须是合法的 JSON 字符串：{e}") from e
                try:
                    obj = json.loads(fixed)
                except json.JSONDecodeError as e2:
                    raise ValidationError(f"final.content 必须是合法的 JSON 字符串：{e2}") from e2

                # Keep the repaired JSON for downstream rendering / writing.
                final["content"] = fixed
            validate_scaffold_plan_schema(obj)
            _require_mentor_deep_dive_in_json(obj, "scaffold")
            # Normalize output to a canonical single-line JSON string.
            final["content"] = json.dumps(obj, ensure_ascii=False)
            return

        if task_kind == "refactor":
            obj = parse_json_content(final)
            validate_refactor_suggestions_schema(obj)
            _require_mentor_deep_dive_in_json(obj, "refactor")
            return

        # Default: no additional validation.
        return

    def _single_turn_fallback(
        self,
        session: SessionState,
        task_kind: str,
        user_text: str,
        tool_results: list[dict[str, Any]],
        emit: EventSink | None,
    ) -> FinalResponse | None:
        """Bypass the tool loop with a fresh single-turn LLM call.

        Uses a minimal, task-specific prompt with code content embedded
        directly.  No JSON envelope requirement, no tool instructions —
        just a simple "here's the code, answer the question" prompt.

        Makes up to 2 attempts with progressively simpler prompts.
        Returns a FinalResponse on success, or None if both fail.
        """
        code_content = _summarize_tool_content(tool_results)
        if not code_content:
            return None

        # Build a simple, task-specific prompt — NO JSON envelope, NO tools.
        task_instruction = _SIMPLE_TASK_PROMPTS.get(task_kind, "请基于以下代码回答用户的问题。")

        # Attempt 1: Task-specific prompt with code.
        system_text = f"{task_instruction}\n\n以下是相关代码：\n\n{code_content}"

        result = self._single_turn_attempt(session, task_kind, user_text, system_text, emit, attempt=1)
        if result is not None:
            return result

        # Attempt 2: Even simpler — bare minimum prompt.
        if emit:
            emit("single_turn_retry", {"reason": "first_attempt_failed", "attempt": 2})

        simple_system = f"你是代码助手。直接用中文回答。\n\n代码：\n{code_content}"

        result = self._single_turn_attempt(session, task_kind, user_text, simple_system, emit, attempt=2)
        return result

    def _single_turn_attempt(
        self,
        session: SessionState,
        task_kind: str,
        user_text: str,
        system_text: str,
        emit: EventSink | None,
        attempt: int = 1,
    ) -> FinalResponse | None:
        """Execute a single-turn LLM call and parse the result."""
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]

        if emit:
            emit("single_turn_llm_call", {"task": task_kind, "system_chars": len(system_text), "user_chars": len(user_text), "attempt": attempt})

        session.stats.llm_calls += 1
        try:
            resp = self.llm.chat(
                messages,
                stream=True,
                think=False,
                request_meta={"task": task_kind, "iter": f"single_turn_{attempt}", "session_id": session.session_id},
                on_event=(lambda p: emit(p["event"], p["data"])) if emit else None,
            )
        except Exception as exc:
            if emit:
                emit("single_turn_error", {"error": str(exc), "attempt": attempt})
            return None

        raw = resp.content.strip()
        if emit:
            emit("single_turn_response", {"chars": len(raw), "raw_content": raw[:2000], "attempt": attempt})

        if not raw:
            return None

        # Try to parse as a proper envelope first.
        try:
            envelopes = parse_envelopes(raw)
            for env in envelopes:
                if env["kind"] == "final":
                    if emit:
                        emit("final_ok", {"task": task_kind, "format": env["format"], "chars": len(env["content"])})
                    return env
        except EnvelopeParseError:
            pass

        # Model returned non-envelope text — wrap it as markdown final.
        if emit:
            emit("final_ok", {"task": task_kind, "format": "markdown", "chars": len(raw)})
        return {"kind": "final", "format": "markdown", "content": raw}


def _build_fallback_response(tool_results: list[dict[str, Any]], reason: str) -> FinalResponse:
    """Synthesize a friendly FinalResponse when the model fails to produce one."""
    if tool_results:
        summary = _summarize_tool_data(tool_results)
        content = f"模型在处理过程中遇到了困难，以下是已收集到的信息：\n\n{summary}\n\n> 提示：可以尝试简化问题或缩小文件范围后重试。"
    else:
        reasons = {
            "empty": "模型返回了空内容",
            "parse_error": "模型输出格式不正确",
            "validation_error": "模型输出未通过验证",
            "max_iters": "已达最大循环次数",
        }
        hint = reasons.get(reason, "未知原因")
        content = f"抱歉，{hint}，未能完成任务。\n\n> 提示：可以尝试简化问题、缩小文件范围，或换一种方式描述需求后重试。"
    return {"kind": "final", "format": "markdown", "content": content}


def _summarize_tool_content(results: list[dict[str, Any]]) -> str:
    """Extract actual file content from tool results for retry injection.

    Unlike _summarize_tool_data (which produces user-facing emoji summaries),
    this returns the raw code/data so the model can analyze it without needing
    to call tools again.  Content is capped to avoid blowing up the context.

    赛题③ 错误处理-超长上下文：动态截断与提示词压缩，工具结果注入上限 4000 字。
    """
    parts: list[str] = []
    total_chars = 0
    _MAX_SUMMARY_CHARS = 4000

    for r in results:
        if total_chars >= _MAX_SUMMARY_CHARS:
            break
        tool = r.get("tool", "")
        ok = r.get("ok", False)
        data = r.get("data") or {}
        if not isinstance(data, dict):
            continue

        if tool == "open_file" and ok:
            path = data.get("path", "?")
            content = data.get("content", "")
            if content:
                remaining = _MAX_SUMMARY_CHARS - total_chars
                snippet = content[:remaining]
                parts.append(f"[文件: {path}]\n{snippet}")
                total_chars += len(snippet) + len(path) + 10
        elif tool == "search_code" and ok:
            matches = data.get("matches", [])
            if isinstance(matches, list):
                lines: list[str] = []
                for m in matches[:10]:
                    if isinstance(m, dict):
                        lines.append(f"  {m.get('path', '?')}:{m.get('line', '?')} {m.get('text', '')}")
                text = "\n".join(lines)
                remaining = _MAX_SUMMARY_CHARS - total_chars
                parts.append(f"[搜索结果]\n{text[:remaining]}")
                total_chars += min(len(text), remaining) + 15
        elif tool == "run_task":
            stdout = data.get("stdout", "")
            stderr = data.get("stderr", "")
            output = stdout or stderr
            if output:
                remaining = _MAX_SUMMARY_CHARS - total_chars
                parts.append(f"[任务输出: {data.get('task', '?')}]\n{output[:remaining]}")
                total_chars += min(len(output), remaining) + 20

    return "\n\n".join(parts)


def _summarize_tool_data(results: list[dict[str, Any]]) -> str:
    """Build a markdown summary from collected tool results."""
    parts: list[str] = []
    for r in results:
        tool = r.get("tool", "unknown")
        ok = r.get("ok", False)
        data = r.get("data") or {}
        if not isinstance(data, dict):
            data = {}

        if tool == "open_file" and ok:
            path = data.get("path", "?")
            content = data.get("content", "")
            lines = content.count("\n") + 1 if content else 0
            parts.append(f"- 📄 已读取 `{path}`（{lines} 行）")
        elif tool == "search_code" and ok:
            matches = data.get("matches", [])
            n = len(matches) if isinstance(matches, list) else 0
            parts.append(f"- 🔍 搜索完成，找到 {n} 处匹配")
        elif tool == "run_task" and ok:
            task = data.get("task", "?")
            exit_code = data.get("exit_code", "?")
            parts.append(f"- ▶️ 执行 `{task}`，退出码 {exit_code}")
        elif tool == "write_files" and ok:
            written = data.get("written", [])
            n = len(written) if isinstance(written, list) else 0
            parts.append(f"- ✏️ 写入 {n} 个文件")
        elif not ok:
            err = r.get("error", {})
            msg = err.get("message", "未知错误") if isinstance(err, dict) else str(err)
            parts.append(f"- ❌ `{tool}` 失败：{msg}")
        else:
            parts.append(f"- ✅ `{tool}` 执行成功")

    return "\n".join(parts) if parts else "（无工具执行记录）"


def _safe_args_preview(args: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in args.items():
        if isinstance(v, str):
            out[k] = v[:200]
        elif isinstance(v, (int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = "<complex>"
    return out


_MAX_TOOL_RESULT_HISTORY_CHARS = 1500


def _truncate_tool_result_for_history(result: dict[str, Any]) -> str:
    """Serialize a tool result for session history, truncating large payloads.

    open_file can return hundreds of lines of code; search_code can return
    dozens of matches.  Storing the full result bloats the context window
    and can cause the model to choke on subsequent rounds.
    """
    data = result.get("data")
    if isinstance(data, dict):
        trimmed = dict(result)
        trimmed_data = dict(data)

        # Truncate open_file content.
        content = trimmed_data.get("content")
        if isinstance(content, str) and len(content) > _MAX_TOOL_RESULT_HISTORY_CHARS:
            trimmed_data["content"] = content[:_MAX_TOOL_RESULT_HISTORY_CHARS] + "\n...[截断]..."
            trimmed_data["truncated"] = True

        # Truncate search_code matches.
        matches = trimmed_data.get("matches")
        if isinstance(matches, list) and len(matches) > 15:
            trimmed_data["matches"] = matches[:15]
            trimmed_data["matches_truncated_from"] = len(matches)

        # Truncate stdout/stderr from run_task.
        for key in ("stdout", "stderr"):
            val = trimmed_data.get(key)
            if isinstance(val, str) and len(val) > _MAX_TOOL_RESULT_HISTORY_CHARS:
                trimmed_data[key] = val[:_MAX_TOOL_RESULT_HISTORY_CHARS] + "\n...[截断]..."

        trimmed["data"] = trimmed_data
        return json.dumps(trimmed, ensure_ascii=False)

    return json.dumps(result, ensure_ascii=False)


def _tool_result_preview(res: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(res, dict):
        return {}
    tool = res.get("tool")
    data = res.get("data") or {}
    if not isinstance(data, dict):
        data = {}

    if tool == "search_code":
        matches = data.get("matches")
        n = len(matches) if isinstance(matches, list) else 0
        return {"matches": n}
    if tool == "open_file":
        return {"path": data.get("path"), "start_line": data.get("start_line"), "end_line": data.get("end_line")}
    if tool == "run_task":
        return {"task": data.get("task"), "exit_code": data.get("exit_code"), "stdout_tail": _tail(data.get("stdout")), "stderr_tail": _tail(data.get("stderr"))}
    if tool == "git_apply_check":
        return {"ok_to_apply": data.get("ok_to_apply"), "stderr_tail": _tail(data.get("stderr"))}
    if tool == "write_files":
        written = data.get("written")
        skipped = data.get("skipped")
        errors = data.get("errors")
        return {
            "dry_run": data.get("dry_run"),
            "written": len(written) if isinstance(written, list) else 0,
            "skipped": len(skipped) if isinstance(skipped, list) else 0,
            "errors": len(errors) if isinstance(errors, list) else 0,
        }
    return {}


def _tail(v: Any, max_chars: int = 600) -> str:
    if not isinstance(v, str):
        return ""
    if len(v) <= max_chars:
        return v
    return "...\n" + v[-max_chars:]
