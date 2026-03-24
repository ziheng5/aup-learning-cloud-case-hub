from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path

from ..config import RagConfig


@dataclass(frozen=True)
class ScannedFile:
    path: str  # repo-relative posix path
    size_bytes: int


class RepoScanner:
    def __init__(self, config: RagConfig) -> None:
        self._config = config

    def scan(self, repo_root: Path) -> list[ScannedFile]:
        repo_root = repo_root.resolve()
        out: list[ScannedFile] = []
        ignore_dirs_set = frozenset(self._config.ignore_dirs)  # O(1) membership per dir
        for root, dirs, filenames in os.walk(repo_root):
            # 赛题/RAG：只扫描项目本体，忽略 .git/.venv/__pycache__/node_modules/.jupypilot（见 config.RagConfig.ignore_dirs）。
            dirs[:] = [d for d in dirs if d not in ignore_dirs_set]
            root_path = Path(root)
            for name in filenames:
                full = root_path / name
                try:
                    st = full.stat()
                except OSError:
                    continue
                if st.st_size > self._config.max_file_bytes:
                    continue
                rel = full.relative_to(repo_root).as_posix()
                if any(fnmatch.fnmatch(rel, g) for g in self._config.ignore_globs):
                    continue
                out.append(ScannedFile(path=rel, size_bytes=int(st.st_size)))
        return out

