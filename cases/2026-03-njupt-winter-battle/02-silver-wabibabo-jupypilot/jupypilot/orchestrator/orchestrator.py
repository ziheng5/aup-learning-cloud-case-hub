from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable

from ..config import Config
from ..llm.ollama_client import OllamaClient
from ..rag.packager import ContextPackager
from ..rag.retriever import HybridRetriever
from ..storage.artifacts import ArtifactsStore
from ..storage.event_log import EventLogger
from ..tools.runtime import ToolRuntime
from ..types import FinalResponse, SessionState
from .context_builder import ContextBuilder
from .memory import MemoryManager
from .policy_guard import PolicyGuard, PolicyViolation
from .prompt_registry import PromptRegistry
from .session import new_session, session_to_dict
from .tool_loop import MaxToolItersError, ToolLoopEngine
from .validator import EnvelopeParseError, extract_single_diff_block


UIEventSink = Callable[[dict[str, Any]], None]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Orchestrator:
    def __init__(self, config: Config, *, prompts: PromptRegistry | None = None) -> None:
        self._config = config
        self._repo_root = config.repo_root()
        self._prompts = prompts or PromptRegistry()

        self._artifacts = ArtifactsStore(repo_root=self._repo_root, artifacts_root=config.artifacts_root())

        # Session-scoped resources
        self._loggers: dict[str, EventLogger] = {}

        # Shared dependencies
        self._llm = OllamaClient(config.ollama, on_event=None)
        self._tools = ToolRuntime(config)
        self._policy = PolicyGuard(config)
        self._context_builder = ContextBuilder(config=config, prompts=self._prompts)
        self._memory = MemoryManager(config=config, llm=self._llm, prompts=self._prompts)
        self._tool_loop = ToolLoopEngine(
            config=config,
            llm=self._llm,
            tools=self._tools,
            policy=self._policy,
            context_builder=self._context_builder,
            memory=self._memory,
        )
        self._retriever = HybridRetriever(config)
        self._packager = ContextPackager()

    @property
    def config(self) -> Config:
        return self._config

    @property
    def artifacts(self) -> ArtifactsStore:
        return self._artifacts

    def start_session(self, repo_path: str | None = None, *, pinned_requirements: list[str] | None = None) -> SessionState:
        repo = str(repo_path or self._repo_root)
        session = new_session(repo_path=repo, pinned_requirements=pinned_requirements)
        session.flags["write_enabled"] = False
        self._loggers[session.session_id] = EventLogger(
            session_id=session.session_id,
            log_path=self._artifacts.layout.logs / f"events_{session.session_id}.jsonl",
        )
        self._emit(session, "session_start", {"repo_path": repo, "artifacts_root": str(self._artifacts.layout.root)})
        return session

    def set_write_enabled(self, session: SessionState, enabled: bool) -> None:
        session.flags["write_enabled"] = bool(enabled)
        self._emit(session, "write_toggle", {"write_enabled": bool(enabled)})

    def pin_requirement(self, session: SessionState, text: str) -> None:
        t = (text or "").strip()
        if not t:
            return
        if t not in session.pinned_requirements:
            session.pinned_requirements.append(t)
            self._emit(session, "pinned_requirement", {"text": t})

    def handle_user_request(
        self,
        session: SessionState,
        *,
        task_kind: str,
        user_text: str,
        on_event: UIEventSink | None = None,
    ) -> FinalResponse:
        """
        Main entry point for UI panels.
        """
        def emit(event: str, data: dict[str, Any]) -> None:
            self._emit(session, event, data)
            if on_event:
                on_event({"ts": _utc_now_iso(), "session_id": session.session_id, "event": event, "data": data})

        emit("user_request", {"task_kind": task_kind, "chars": len(user_text)})

        # Code-related tasks: build retrieved context.
        retrieved_context = ""
        if task_kind in {"code_qa", "code_patch", "testgen", "refactor", "refactor_diff"}:
            emit("rag_retrieve_start", {"topk": self._config.context.retrieved_topk})
            ctx = self._retriever.retrieve(self._repo_root, user_text, topk=self._config.context.retrieved_topk)
            packaged = self._packager.package(ctx)
            retrieved_context = packaged.retrieved_context
            emit("retrieved_context", {"items": len(packaged.items), "chars": len(retrieved_context)})
            emit("rag_retrieve_end", {"items": len(packaged.items), "chars": len(retrieved_context)})

        try:
            final = self._tool_loop.run(
                session,
                task_kind=task_kind,
                user_text=user_text,
                retrieved_context=retrieved_context,
                emit=emit,
            )
        except PolicyViolation as e:
            self._emit(session, "policy_block", {"code": e.code, "message": str(e)}, level="WARN")
            return {"kind": "final", "format": "markdown", "content": f"请求被策略拒绝：{e}\n\n请改写为正常工程任务描述后重试。"}
        except (EnvelopeParseError, MaxToolItersError) as e:
            self._emit(session, "tool_loop_error", {"error": type(e).__name__, "message": str(e)}, level="WARN")
            return {"kind": "final", "format": "markdown", "content": f"处理过程中出现问题：{e}\n\n> 提示：可以尝试简化问题或缩小文件范围后重试。"}

        # Persist artifacts for certain tasks.
        self._persist_artifacts(session, task_kind, final)
        self._persist_session_snapshot(session)
        return final

    # --------------------------
    # Verify (tool-only) helpers
    # --------------------------
    def run_verify(self, session: SessionState, task: str) -> dict[str, Any]:
        self._emit(session, "verify_start", {"task": task})
        res = self._tools.execute("run_task", {"task": task}, session)
        session.stats.tool_calls += 1
        session.history.append({"role": "assistant", "content": json.dumps(res, ensure_ascii=False)})
        self._emit(session, "verify_end", {"task": task, "ok": res.get("ok"), "data": res.get("data")})
        self._persist_session_snapshot(session)
        return res

    def write_files(self, session: SessionState, plan: dict[str, Any], *, dry_run: bool = True, overwrite: bool = False) -> dict[str, Any]:
        args = {"plan": plan, "dry_run": dry_run, "overwrite": overwrite}
        res = self._tools.execute("write_files", args, session)
        session.stats.tool_calls += 1
        session.history.append({"role": "assistant", "content": json.dumps(res, ensure_ascii=False)})
        self._emit(session, "write_files", {"ok": res.get("ok"), "data": res.get("data"), "error": res.get("error")})
        self._persist_session_snapshot(session)
        return res

    # --------------------------
    # Persistence helpers
    # --------------------------
    def _persist_artifacts(self, session: SessionState, task_kind: str, final: FinalResponse) -> None:
        try:
            if task_kind in {"code_patch", "refactor_diff"} and final["format"] == "markdown":
                diff = extract_single_diff_block(final["content"])
                path = self._artifacts.write_patch(session_id=session.session_id, diff_text=diff, label=task_kind)
                self._emit(session, "artifact_written", {"kind": "patch", "path": str(path)})
                session.artifacts.setdefault("patches", []).append(
                    {"id": path.name, "path": str(path), "created_at": _utc_now_iso(), "kind": "patch"}
                )
        except Exception as e:
            self._emit(session, "artifact_error", {"error": str(e)}, level="WARN")

    def _persist_session_snapshot(self, session: SessionState) -> None:
        try:
            snap = session_to_dict(session)
            path = self._artifacts.write_session_snapshot(session_id=session.session_id, obj=snap)
            self._emit(session, "session_snapshot", {"path": str(path)})
        except Exception as e:
            self._emit(session, "session_snapshot_error", {"error": str(e)}, level="WARN")

    # --------------------------
    # Event emission
    # --------------------------
    def _emit(self, session: SessionState, event: str, data: dict[str, Any], *, level: str = "INFO") -> None:
        logger = self._loggers.get(session.session_id)
        if logger:
            logger.emit(event, level=level, data=data)

    def get_recent_events(self, session: SessionState, *, level: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        logger = self._loggers.get(session.session_id)
        if not logger:
            return []
        return logger.list_recent(level=level, limit=limit)
