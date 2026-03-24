from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Event:
    ts: str
    session_id: str
    seq: int
    event: str
    level: str
    data: dict[str, Any]
    duration_ms: int | None = None


class EventLogger:
    def __init__(self, *, session_id: str, log_path: Path, keep_last: int = 500) -> None:
        self._session_id = session_id
        self._log_path = log_path
        self._keep_last = keep_last
        self._seq = 0
        # deque(maxlen) avoids O(keep_last) copy on every trim when exceeding capacity.
        self._events: deque[Event] = deque(maxlen=keep_last)
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Best-effort: keep events in memory if filesystem is not writable.
            pass

    @property
    def log_path(self) -> Path:
        return self._log_path

    def emit(self, event: str, *, level: str = "INFO", data: dict[str, Any] | None = None, duration_ms: int | None = None) -> Event:
        self._seq += 1
        e = Event(
            ts=_utc_now_iso(),
            session_id=self._session_id,
            seq=self._seq,
            event=event,
            level=level,
            data=data or {},
            duration_ms=duration_ms,
        )
        self._events.append(e)
        self._append_jsonl(e)
        return e

    def list_recent(self, *, level: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        items = list(self._events)
        if level:
            items = [e for e in items if e.level == level]
        items = items[-limit:]
        return [self._as_dict(e) for e in items]

    def _append_jsonl(self, e: Event) -> None:
        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(self._as_dict(e), ensure_ascii=False) + "\n")
        except OSError:
            # Logging must never break the main flow.
            return

    @staticmethod
    def _as_dict(e: Event) -> dict[str, Any]:
        d: dict[str, Any] = {
            "ts": e.ts,
            "session_id": e.session_id,
            "seq": e.seq,
            "event": e.event,
            "level": e.level,
            "data": e.data,
        }
        if e.duration_ms is not None:
            d["duration_ms"] = e.duration_ms
        return d
