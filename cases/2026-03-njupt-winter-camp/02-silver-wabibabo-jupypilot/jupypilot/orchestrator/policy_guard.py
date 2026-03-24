from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..config import Config
from ..types import SessionState, ToolCall


class PolicyViolation(RuntimeError):
    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


# 赛题③ 错误处理-输入合法性：拦截「ignore system prompt」等注入模式，拒绝越权工具调用。详见 赛题符合性-实现映射.md。
_INJECTION_PATTERNS = [
    r"ignore\s+system\s+prompt",
    r"reveal\s+system\s+prompt",
    r"泄露提示词",
    r"越权",
    r"执行任意命令",
    r"下载运行",
]


def _looks_like_path_escape(p: str) -> bool:
    s = p.replace("\\", "/")
    if s.startswith("/") or re.match(r"^[a-zA-Z]:/", s):
        return True
    parts = [x for x in s.split("/") if x and x != "."]
    up = 0
    for part in parts:
        if part == "..":
            up += 1
        else:
            up = max(0, up - 1)
    return ".." in parts


@dataclass(frozen=True)
class PolicyGuard:
    config: Config

    def check_user_input(self, user_text: str) -> None:
        t = (user_text or "").lower()
        for pat in _INJECTION_PATTERNS:
            if re.search(pat, t, flags=re.IGNORECASE):
                raise PolicyViolation("E_POLICY", "疑似提示词注入内容，请改写你的请求后重试")

    def check_tool_call(self, call: ToolCall, session: SessionState) -> None:
        tool = call["tool"]
        args = call.get("args") or {}
        if tool == "run_task":
            task = args.get("task")
            if task not in ("ruff_check", "pytest_q"):
                raise PolicyViolation("E_POLICY", "不允许执行该任务", details={"allowed": ["ruff_check", "pytest_q"]})
            return

        if tool == "open_file":
            path = args.get("path")
            if not isinstance(path, str) or not path.strip():
                raise PolicyViolation("E_VALIDATION", "open_file.path 必须为非空字符串")
            if _looks_like_path_escape(path):
                raise PolicyViolation("E_PATH", "open_file.path 不是安全的相对路径")
            return

        if tool == "search_code":
            query = args.get("query")
            if not isinstance(query, str) or not query.strip():
                raise PolicyViolation("E_VALIDATION", "search_code.query 必须为非空字符串")
            return

        if tool == "git_apply_check":
            if not self.config.tools.allow_git_apply_check:
                raise PolicyViolation("E_POLICY", "配置已禁用 git_apply_check")
            diff = args.get("diff")
            if not isinstance(diff, str) or not diff.strip():
                raise PolicyViolation("E_VALIDATION", "git_apply_check.diff 必须为非空字符串")
            return

        if tool == "write_files":
            if not self.config.tools.allow_write_files:
                raise PolicyViolation("E_POLICY", "配置已禁用 write_files")
            if not bool(session.flags.get("write_enabled", False)):
                raise PolicyViolation("E_POLICY", "写入文件需要在 UI 打开“允许写入文件”开关")
            plan = args.get("plan")
            if not isinstance(plan, dict):
                raise PolicyViolation("E_VALIDATION", "write_files.plan 必须为对象")
            return

        raise PolicyViolation("E_POLICY", f"不允许使用该工具：{tool}")
