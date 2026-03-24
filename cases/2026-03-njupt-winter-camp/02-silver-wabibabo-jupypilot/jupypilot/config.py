from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Mapping


# 赛题② 集群资源利用：集群内网大模型 API 的 endpoint 与模型在此配置。
@dataclass(frozen=True)
class OllamaConfig:
    endpoint: str = "http://open-webui-ollama.open-webui:11434/api/chat"
    model: str = "qwen3-coder:30b"  # 与 Ollama api/tags 返回的 name 一致（冒号格式）
    temperature: float = 0.2
    top_p: float = 0.9
    timeout_s: int = 300
    """总超时（流式单 chunk 与整次请求上限）。"""
    timeout_first_byte_s: int = 120
    """等待首字节超时（如 ROCm 冷启动较慢可适当调大）。"""
    max_retries: int = 3
    backoff_base_s: float = 1.0
    num_ctx: int = 32768
    num_predict: int = 8192
    """单次生成的最大 token 数，避免服务端默认过小导致回答被截断；8192 约可支撑较长代码解释/补丁/测试。"""


@dataclass(frozen=True)
class ToolLoopConfig:
    max_iters: int = 15
    correction_retries: int = 2


@dataclass(frozen=True)
class ToolsConfig:
    allow_git_apply_check: bool = True
    allow_write_files: bool = False


@dataclass(frozen=True)
class LimitsConfig:
    open_file_max_lines: int = 400
    search_max_results: int = 30
    max_stdout_bytes: int = 65536
    max_stderr_bytes: int = 65536
    subprocess_timeout_s: int = 180
    rg_timeout_s: int = 30


# 赛题③ 错误处理-超长上下文：分段滚动(history_max_turns/chunk_lines/retrieved_topk)、级联摘要(memory_summary_trigger_tokens)、动态截断(max_ctx_tokens/completion_budget_tokens)。
@dataclass(frozen=True)
class ContextConfig:
    max_ctx_tokens: int = 32768
    completion_budget_tokens: int = 3072
    retrieved_topk: int = 8
    chunk_lines: int = 300
    history_max_turns: int = 12
    memory_summary_trigger_tokens: int = 24000


# RAG 扫描忽略目录：工程化标准，只对“项目本体”建索引，排除 .git/.venv/__pycache__/node_modules/.jupypilot。赛题符合性见 赛题符合性-实现映射.md。
@dataclass(frozen=True)
class RagConfig:
    ignore_dirs: tuple[str, ...] = (
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".ruff_cache",
        ".jupypilot",
        ".jupypilot_chroma",
    )
    ignore_globs: tuple[str, ...] = (
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.gif",
        "*.pdf",
        "*.zip",
        "*.tar",
        "*.gz",
        "*.bin",
    )
    max_file_bytes: int = 800_000


@dataclass(frozen=True)
class Config:
    repo_path: str = "."
    artifacts_dir: str = ".jupypilot"
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    tool_loop: ToolLoopConfig = field(default_factory=ToolLoopConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    rag: RagConfig = field(default_factory=RagConfig)

    def repo_root(self) -> Path:
        return Path(self.repo_path).expanduser().resolve()

    def artifacts_root(self) -> Path:
        p = Path(self.artifacts_dir).expanduser()
        if p.is_absolute():
            return p
        return self.repo_root() / p


def _try_load_yaml(path: Path) -> dict[str, Any] | None:
    try:
        import yaml  # type: ignore
    except Exception:
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = yaml.safe_load(f) or {}
        if not isinstance(obj, dict):
            raise ValueError("config root must be a mapping")
        return obj
    except FileNotFoundError:
        return None


def _try_load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError("config root must be a mapping")
        return obj
    except FileNotFoundError:
        return None


def _deep_merge(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(dict(out[k]), v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def _as_bool(v: str) -> bool:
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if repo := env.get("JUPYPILOT_REPO_PATH"):
        out["repo_path"] = repo
    if art := env.get("JUPYPILOT_ARTIFACTS_DIR"):
        out["artifacts_dir"] = art
    if endpoint := env.get("JUPYPILOT_OLLAMA_ENDPOINT"):
        out.setdefault("ollama", {})["endpoint"] = endpoint
    if model := env.get("JUPYPILOT_OLLAMA_MODEL"):
        out.setdefault("ollama", {})["model"] = model
    if allow_write := env.get("JUPYPILOT_ALLOW_WRITE_FILES"):
        out.setdefault("tools", {})["allow_write_files"] = _as_bool(allow_write)
    return out


def _ollama_from_dict(d: Mapping[str, Any]) -> OllamaConfig:
    base = OllamaConfig()
    return OllamaConfig(
        endpoint=str(d.get("endpoint", base.endpoint)),
        model=str(d.get("model", base.model)),
        temperature=float(d.get("temperature", base.temperature)),
        top_p=float(d.get("top_p", base.top_p)),
        timeout_s=int(d.get("timeout_s", base.timeout_s)),
        timeout_first_byte_s=int(d.get("timeout_first_byte_s", base.timeout_first_byte_s)),
        max_retries=int(d.get("max_retries", base.max_retries)),
        backoff_base_s=float(d.get("backoff_base_s", base.backoff_base_s)),
        num_ctx=int(d.get("num_ctx", base.num_ctx)),
        num_predict=int(d.get("num_predict", base.num_predict)),
    )


def _tool_loop_from_dict(d: Mapping[str, Any]) -> ToolLoopConfig:
    base = ToolLoopConfig()
    return ToolLoopConfig(
        max_iters=int(d.get("max_iters", base.max_iters)),
        correction_retries=int(d.get("correction_retries", base.correction_retries)),
    )


def _tools_from_dict(d: Mapping[str, Any]) -> ToolsConfig:
    base = ToolsConfig()
    return ToolsConfig(
        allow_git_apply_check=bool(d.get("allow_git_apply_check", base.allow_git_apply_check)),
        allow_write_files=bool(d.get("allow_write_files", base.allow_write_files)),
    )


def _limits_from_dict(d: Mapping[str, Any]) -> LimitsConfig:
    base = LimitsConfig()
    return LimitsConfig(
        open_file_max_lines=int(d.get("open_file_max_lines", base.open_file_max_lines)),
        search_max_results=int(d.get("search_max_results", base.search_max_results)),
        max_stdout_bytes=int(d.get("max_stdout_bytes", base.max_stdout_bytes)),
        max_stderr_bytes=int(d.get("max_stderr_bytes", base.max_stderr_bytes)),
        subprocess_timeout_s=int(d.get("subprocess_timeout_s", base.subprocess_timeout_s)),
        rg_timeout_s=int(d.get("rg_timeout_s", base.rg_timeout_s)),
    )


def _context_from_dict(d: Mapping[str, Any]) -> ContextConfig:
    base = ContextConfig()
    return ContextConfig(
        max_ctx_tokens=int(d.get("max_ctx_tokens", base.max_ctx_tokens)),
        completion_budget_tokens=int(d.get("completion_budget_tokens", base.completion_budget_tokens)),
        retrieved_topk=int(d.get("retrieved_topk", base.retrieved_topk)),
        chunk_lines=int(d.get("chunk_lines", base.chunk_lines)),
        history_max_turns=int(d.get("history_max_turns", base.history_max_turns)),
        memory_summary_trigger_tokens=int(d.get("memory_summary_trigger_tokens", base.memory_summary_trigger_tokens)),
    )


def _rag_from_dict(d: Mapping[str, Any]) -> RagConfig:
    base = RagConfig()
    ignore_dirs = tuple(d.get("ignore_dirs", base.ignore_dirs))
    ignore_globs = tuple(d.get("ignore_globs", base.ignore_globs))
    return RagConfig(
        ignore_dirs=ignore_dirs,
        ignore_globs=ignore_globs,
        max_file_bytes=int(d.get("max_file_bytes", base.max_file_bytes)),
    )


class ConfigLoader:
    def load(
        self,
        config_path: str | None = None,
        overrides: Mapping[str, Any] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> Config:
        """
        Load config with deterministic precedence (low -> high):
        1) defaults
        2) config file (YAML/JSON, optional)
        3) environment variables (optional)
        4) explicit overrides (Notebook)
        """
        data: dict[str, Any] = {}

        if config_path:
            p = Path(config_path)
            loaded = _try_load_yaml(p)
            if loaded is None and p.suffix.lower() == ".json":
                loaded = _try_load_json(p)
            if loaded is not None:
                data = _deep_merge(data, loaded)

        env_map = env if env is not None else os.environ
        data = _deep_merge(data, _env_overrides(env_map))
        if overrides:
            data = _deep_merge(data, overrides)

        base = Config()
        return Config(
            repo_path=str(data.get("repo_path", base.repo_path)),
            artifacts_dir=str(data.get("artifacts_dir", base.artifacts_dir)),
            ollama=_ollama_from_dict(data.get("ollama", {}) or {}),
            tool_loop=_tool_loop_from_dict(data.get("tool_loop", {}) or {}),
            tools=_tools_from_dict(data.get("tools", {}) or {}),
            limits=_limits_from_dict(data.get("limits", {}) or {}),
            context=_context_from_dict(data.get("context", {}) or {}),
            rag=_rag_from_dict(data.get("rag", {}) or {}),
        )


def load_config(config_path: str | None = None, overrides: Mapping[str, Any] | None = None) -> Config:
    return ConfigLoader().load(config_path=config_path, overrides=overrides)
