from __future__ import annotations

import secrets
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from ..types import MemorySummary, SessionState


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_session(*, repo_path: str, pinned_requirements: list[str] | None = None) -> SessionState:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    rnd = secrets.token_hex(4)
    session_id = f"{ts}-{rnd}"
    return SessionState(
        session_id=session_id,
        repo_path=repo_path,
        created_at=utc_now_iso(),
        history=[],
        memory_summary=MemorySummary(),
        pinned_requirements=list(pinned_requirements or []),
        flags={},
    )


def session_to_dict(session: SessionState) -> dict[str, Any]:
    """
    Serialize session to JSON-safe dict (for snapshots).
    """
    d = asdict(session)
    return d

