from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def _safe_int(v: Any) -> int | None:
    try:
        return int(v)
    except Exception:
        return None


def _safe_float(v: Any) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def _fmt_ts_zh(ts: Any) -> str:
    """
    Input: ISO-ish string like 2026-02-19T12:34:56.789+00:00 (UTC)
    Output: 2026-02-19 12:34:56 in local time (avoid showing 'T' or timezone letters)
    """
    if not isinstance(ts, str) or not ts:
        return ""
    try:
        # 解析 ISO 时间；无时区时按 UTC 处理
        s = ts.strip()
        if len(s) < 19:
            return s.replace("T", " ") if "T" in s else s
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local = dt.astimezone()
        return local.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        head = ts[:19].replace("T", " ")
        return head


def level_to_zh(level: Any) -> str:
    if level == "INFO":
        return "信息"
    if level == "WARN":
        return "警告"
    if level == "ERROR":
        return "错误"
    return "信息"


def task_kind_to_zh(task: Any) -> str:
    if not isinstance(task, str) or not task:
        return "未知任务"
    mapping = {
        "code_qa": "代码问答",
        "code_patch": "生成补丁",
        "testgen": "生成测试",
        "refactor": "重构建议",
        "refactor_diff": "重构补丁",
        "scaffold": "脚手架",
        "memory_summary": "历史摘要",
    }
    return mapping.get(task, task)


def tool_to_zh(tool: Any) -> str:
    if not isinstance(tool, str) or not tool:
        return "未知工具"
    mapping = {
        "search_code": "代码检索",
        "open_file": "打开文件",
        "run_task": "运行任务",
        "git_apply_check": "补丁可应用性检查",
        "write_files": "写入文件",
    }
    return mapping.get(tool, tool)


def run_task_to_zh(task: Any) -> str:
    if task == "ruff_check":
        return "代码检查"
    if task == "pytest_q":
        return "运行测试"
    return "运行任务"


def event_to_zh(event: Any) -> str:
    if not isinstance(event, str) or not event:
        return "事件"
    mapping = {
        "session_start": "会话开始",
        "user_request": "用户请求",
        "pinned_requirement": "固定约束",
        "write_toggle": "写入开关",
        "rag_retrieve_start": "检索开始",
        "retrieved_context": "检索结果",
        "rag_retrieve_end": "检索结束",
        "tool_loop_iter": "循环迭代",
        "memory_summary_start": "历史摘要开始",
        "memory_summary_end": "历史摘要结束",
        "llm_request_start": "模型请求开始",
        "llm_retry": "模型重试",
        "llm_usage": "词元统计",
        "llm_request_end": "模型请求结束",
        "llm_response": "模型输出",
        "llm_output_invalid": "模型输出不合规",
        "tool_call": "工具调用",
        "tool_result": "工具结果",
        "final_ok": "最终输出完成",
        "policy_block": "策略拦截",
        "artifact_written": "产物写入",
        "artifact_error": "产物写入失败",
        "session_snapshot": "会话快照",
        "session_snapshot_error": "会话快照失败",
        "verify_start": "校验开始",
        "verify_end": "校验结束",
        "write_files": "写文件",
    }
    return mapping.get(event, "事件")


def _bool_zh(v: Any) -> str:
    return "是" if bool(v) else "否"


def format_event_line_zh(e: Mapping[str, Any]) -> str:
    """
    Convert internal event payload to a Chinese, demo-friendly single line.

    Expected shapes:
    - UI callback: {"ts": "...", "event": "...", "data": {...}, ...}
    - EventLogger: {"ts": "...", "level": "INFO|WARN|ERROR", "event": "...", "data": {...}, ...}
    """
    event = e.get("event")
    data = e.get("data") if isinstance(e.get("data"), dict) else {}
    level = e.get("level")

    ts = _fmt_ts_zh(e.get("ts"))
    ts_part = f"{ts} " if ts else ""
    level_part = f"【{level_to_zh(level)}】" if isinstance(level, str) and level else ""

    title = event_to_zh(event)
    details = ""

    if event == "user_request":
        task = task_kind_to_zh(data.get("task_kind"))
        chars = _safe_int(data.get("chars"))
        details = f"：{task}"
        if isinstance(chars, int):
            details += f"，{chars} 字符"

    elif event in {"rag_retrieve_start"}:
        topk = _safe_int(data.get("topk"))
        if isinstance(topk, int):
            details = f"：最多 {topk} 条"

    elif event in {"retrieved_context", "rag_retrieve_end"}:
        items = _safe_int(data.get("items"))
        chars = _safe_int(data.get("chars"))
        parts: list[str] = []
        if isinstance(items, int):
            parts.append(f"{items} 条")
        if isinstance(chars, int):
            parts.append(f"{chars} 字符")
        if parts:
            details = "：" + "，".join(parts)

    elif event == "tool_loop_iter":
        it = _safe_int(data.get("iter"))
        task = task_kind_to_zh(data.get("task"))
        if isinstance(it, int):
            details = f"：第 {it + 1} 轮，{task}"
        else:
            details = f"：{task}"

    elif event == "memory_summary_start":
        msgs = _safe_int(data.get("messages"))
        est = _safe_int(data.get("estimated_tokens"))
        parts = []
        if isinstance(msgs, int):
            parts.append(f"{msgs} 条消息")
        if isinstance(est, int):
            parts.append(f"约 {est} 词元")
        if parts:
            details = "：" + "，".join(parts)

    elif event == "memory_summary_end":
        ok = data.get("ok")
        kept = _safe_int(data.get("kept_messages"))
        details = f"：成功={_bool_zh(ok)}"
        if isinstance(kept, int):
            details += f"，保留 {kept} 条"

    elif event == "llm_request_start":
        attempt = _safe_int(data.get("attempt"))
        meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
        task = task_kind_to_zh(meta.get("task"))
        stream = data.get("stream")
        parts = [task]
        if isinstance(attempt, int):
            parts.append(f"第 {attempt + 1} 次尝试")
        parts.append(f"流式={_bool_zh(stream)}")
        details = "：" + "，".join([p for p in parts if p])

    elif event == "llm_retry":
        hint = str(data.get("hint_zh") or "").strip() or "集群接口异常，正在重试..."
        attempt = _safe_int(data.get("attempt"))
        max_retries = _safe_int(data.get("max_retries"))
        wait_s = _safe_float(data.get("wait_s"))
        duration_ms = _safe_int(data.get("duration_ms"))
        retry_part = ""
        if isinstance(attempt, int) and isinstance(max_retries, int) and max_retries > 0:
            retry_part = f"（{attempt + 1}/{max_retries}）"
        elif isinstance(attempt, int):
            retry_part = f"（{attempt + 1}）"
        # 区分“本次请求已耗时”（如超时/连接失败）与“退避等待时间”，避免用户误以为只等了 wait_s
        duration_part = ""
        if isinstance(duration_ms, int) and duration_ms > 0:
            duration_part = f"，本次请求已耗时 {duration_ms / 1000:.1f} 秒"
        wait_part = f"，退避等待 {wait_s:.1f} 秒" if isinstance(wait_s, float) and wait_s > 0 else ""
        details = f"：{hint}{retry_part}{duration_part}{wait_part}"

    elif event == "llm_usage":
        pt = _safe_int(data.get("prompt_tokens"))
        ct = _safe_int(data.get("completion_tokens"))
        tt = _safe_int(data.get("total_tokens"))
        parts = []
        if isinstance(tt, int):
            parts.append(f"总 {tt}")
        if isinstance(pt, int):
            parts.append(f"输入 {pt}")
        if isinstance(ct, int):
            parts.append(f"输出 {ct}")
        if parts:
            details = "：" + "，".join(parts)

    elif event == "llm_request_end":
        ok = data.get("ok")
        dur = _safe_int(data.get("duration_ms"))
        chars = _safe_int(data.get("chars"))
        details = f"：成功={_bool_zh(ok)}"
        if isinstance(dur, int):
            details += f"，耗时 {dur} 毫秒"
        if isinstance(chars, int):
            details += f"，{chars} 字符"

    elif event == "llm_response":
        chars = _safe_int(data.get("chars"))
        raw = data.get("raw_content")
        if isinstance(chars, int):
            details = f"：{chars} 字符"
        if isinstance(raw, str) and raw.strip():
            preview = raw.strip()[:500].replace("\n", "\\n")
            details += f"\n    原始输出：{preview}"

    elif event == "llm_output_invalid":
        reason = data.get("reason")
        left = _safe_int(data.get("correction_left"))
        raw = data.get("raw_content")
        details = f"：{reason}"
        if isinstance(left, int):
            details += f"，剩余重试 {left} 次"
        if isinstance(raw, str) and raw.strip():
            preview = raw.strip()[:500].replace("\n", "\\n")
            details += f"\n    原始输出：{preview}"

    elif event == "tool_call":
        tool = tool_to_zh(data.get("tool"))
        args = data.get("args") if isinstance(data.get("args"), dict) else {}
        details = f"：{tool}"
        if tool == "打开文件":
            path = args.get("path")
            s = _safe_int(args.get("start_line"))
            ed = _safe_int(args.get("end_line"))
            if isinstance(path, str) and isinstance(s, int) and isinstance(ed, int):
                details += f"（{path}，{s}-{ed} 行）"
        elif tool == "代码检索":
            q = args.get("query")
            if isinstance(q, str) and q.strip():
                details += f"（关键词：{q.strip()[:60]}）"
        elif tool == "运行任务":
            t = run_task_to_zh(args.get("task"))
            details += f"（{t}）"

    elif event == "tool_result":
        tool = tool_to_zh(data.get("tool"))
        ok = data.get("ok")
        details = f"：{tool}，成功={_bool_zh(ok)}"

    elif event == "final_ok":
        task = task_kind_to_zh(data.get("task"))
        chars = _safe_int(data.get("chars"))
        details = f"：{task}"
        if isinstance(chars, int):
            details += f"，{chars} 字符"

    elif event == "verify_start":
        details = f"：{run_task_to_zh(data.get('task'))}"

    elif event == "verify_end":
        ok = data.get("ok")
        details = f"：{run_task_to_zh(data.get('task'))}，成功={_bool_zh(ok)}"

    elif event == "write_toggle":
        details = f"：允许写入={_bool_zh(data.get('write_enabled'))}"

    elif event == "pinned_requirement":
        text = data.get("text")
        if isinstance(text, str) and text.strip():
            details = f"：{text.strip()}"

    return f"{ts_part}{level_part}{title}{details}".strip()
