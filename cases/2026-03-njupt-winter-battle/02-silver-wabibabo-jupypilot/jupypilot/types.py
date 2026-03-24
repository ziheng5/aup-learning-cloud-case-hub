from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


Role = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    role: Role
    content: str


ToolName = Literal["search_code", "open_file", "run_task", "git_apply_check", "write_files"]


class ToolCall(TypedDict):
    kind: Literal["tool"]
    tool: ToolName
    args: dict[str, Any]


class FinalResponse(TypedDict):
    kind: Literal["final"]
    format: Literal["markdown", "json"]
    content: str


Envelope = ToolCall | FinalResponse


class ToolError(TypedDict, total=False):
    code: str
    message: str
    details: dict[str, Any]


class ToolResult(TypedDict):
    ok: bool
    tool: str
    data: dict[str, Any] | None
    error: ToolError | None


@dataclass
class MemorySummary:
    constraints: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    progress: list[str] = field(default_factory=list)
    todo: list[str] = field(default_factory=list)
    pitfalls: list[str] = field(default_factory=list)
    context: str = ""


@dataclass
class ArtifactRef:
    id: str
    path: str
    created_at: str
    kind: str


@dataclass
class SessionStats:
    llm_calls: int = 0
    tool_calls: int = 0
    tool_iters: int = 0


@dataclass
class SessionState:
    session_id: str
    repo_path: str
    created_at: str
    history: list[ChatMessage] = field(default_factory=list)
    memory_summary: MemorySummary = field(default_factory=MemorySummary)
    pinned_requirements: list[str] = field(default_factory=list)
    flags: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, list[ArtifactRef]] = field(default_factory=lambda: {"patches": [], "reports": [], "sessions": []})
    stats: SessionStats = field(default_factory=SessionStats)
