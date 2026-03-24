from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_hash(repo_root: Path) -> str:
    h = hashlib.sha1(str(repo_root).encode("utf-8")).hexdigest()
    return h[:10]


@dataclass(frozen=True)
class ArtifactsLayout:
    root: Path
    logs: Path
    sessions: Path
    patches: Path
    reports: Path
    scaffold: Path
    index: Path


class ArtifactsStore:
    def __init__(self, *, repo_root: Path, artifacts_root: Path) -> None:
        self._repo_root = repo_root.resolve()
        self._root = self._ensure_root(artifacts_root)
        self.layout = ArtifactsLayout(
            root=self._root,
            logs=self._root / "logs",
            sessions=self._root / "sessions",
            patches=self._root / "patches",
            reports=self._root / "reports",
            scaffold=self._root / "scaffold",
            index=self._root / "index",
        )
        self._ensure_dirs()

    def _ensure_root(self, preferred: Path) -> Path:
        preferred = preferred.expanduser()
        if not preferred.is_absolute():
            preferred = (self._repo_root / preferred).resolve()
        try:
            preferred.mkdir(parents=True, exist_ok=True)
            return preferred
        except OSError:
            pass

        # Fallback to user cache directory.
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache")
        root = (base / "jupypilot" / _repo_hash(self._repo_root)).resolve()
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError:
            # As a last resort, keep using preferred path in a best-effort mode.
            return preferred
        return root

    def _ensure_dirs(self) -> None:
        for p in (
            self.layout.logs,
            self.layout.sessions,
            self.layout.patches,
            self.layout.reports,
            self.layout.scaffold,
            self.layout.index,
        ):
            try:
                p.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue

    def write_patch(self, *, session_id: str, diff_text: str, label: str = "patch") -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = self.layout.patches / f"{label}_{session_id}_{ts}.diff"
        path.write_text(diff_text, encoding="utf-8")
        return path

    def write_report(self, *, session_id: str, obj: Any, label: str = "report") -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = self.layout.reports / f"{label}_{session_id}_{ts}.json"
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def write_session_snapshot(self, *, session_id: str, obj: Any) -> Path:
        path = self.layout.sessions / f"session_{session_id}.json"
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
