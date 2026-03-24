from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TextChunk:
    path: str
    start_line: int
    end_line: int
    text: str


class LineChunker:
    def __init__(self, *, chunk_lines: int) -> None:
        self._chunk_lines = max(50, int(chunk_lines))

    def chunk_file(self, repo_root: Path, relpath: str) -> list[TextChunk]:
        full = (repo_root / relpath).resolve()
        try:
            content = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        lines = content.splitlines()
        chunks: list[TextChunk] = []
        n = len(lines)
        i = 0
        while i < n:
            start = i + 1
            end = min(n, i + self._chunk_lines)
            text = "\n".join(lines[i:end])
            chunks.append(TextChunk(path=relpath, start_line=start, end_line=end, text=text))
            i = end
        return chunks

