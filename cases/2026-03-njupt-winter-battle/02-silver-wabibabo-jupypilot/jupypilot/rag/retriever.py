from __future__ import annotations

import heapq
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Config
from ..tools.subprocess_runner import SubprocessRunner, SubprocessError
from .vector_store import CHROMA_PERSIST_DIR, ChromaVectorStore, VectorHit


@dataclass(frozen=True)
class RetrievedContext:
    path: str
    start_line: int
    end_line: int
    snippet: str
    score: float


_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")


def extract_candidate_queries(text: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    # Prefer identifiers; fall back to truncated text.
    idents = list(dict.fromkeys(_IDENT_RE.findall(t)))  # stable unique
    queries: list[str] = []
    queries.extend(idents[:12])
    if t not in queries:
        queries.append(t[:200])
    return queries


class KeywordRetriever:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._runner = SubprocessRunner(
            max_stdout_bytes=config.limits.max_stdout_bytes,
            max_stderr_bytes=config.limits.max_stderr_bytes,
        )

    def retrieve(self, repo_root: Path, query: str, *, topk: int | None = None) -> list[RetrievedContext]:
        repo_root = repo_root.resolve()
        topk = topk or self._config.context.retrieved_topk
        candidates = extract_candidate_queries(query)
        # path -> hit lines (set avoids duplicate line numbers from multiple queries, O(1) add).
        scored: dict[str, set[int]] = {}

        for q in candidates:
            hits = self._rg_search(repo_root, q)
            for h in hits:
                scored.setdefault(h["path"], set()).add(h["line"])

        if not scored:
            return []

        # Score files by hit count; prefer core/tests heuristics.
        file_scores: list[tuple[str, float]] = []
        for path, lines in scored.items():
            s = float(len(lines))
            p = path.replace("\\", "/")
            if "/tests/" in f"/{p}/" or p.startswith("tests/"):
                s += 0.5
            if "/core/" in f"/{p}/" or p.startswith("core/"):
                s += 0.3
            file_scores.append((path, s))
        # Top-k only: O(n log k) instead of full sort O(n log n).
        top_files = heapq.nlargest(max(3, topk), file_scores, key=lambda x: x[1])

        contexts: list[RetrievedContext] = []
        for path, s in top_files:
            lines_for_path = scored[path]
            # When many lines, avoid full sort: take smallest 50 then sort for range merge. O(n log 50).
            if len(lines_for_path) <= 50:
                unique_lines = sorted(lines_for_path)
            else:
                unique_lines = sorted(heapq.nsmallest(50, lines_for_path))
            ranges = _merge_line_windows(unique_lines, window=40, max_lines=self._config.limits.open_file_max_lines)
            for start_line, end_line in ranges:
                snippet = _read_lines(repo_root / path, start_line, end_line)
                if not snippet:
                    continue
                contexts.append(
                    RetrievedContext(
                        path=path,
                        start_line=start_line,
                        end_line=end_line,
                        snippet=snippet,
                        score=s,
                    )
                )

        # Return top-k by score without full sort when topk is small.
        if len(contexts) <= topk:
            contexts.sort(key=lambda c: c.score, reverse=True)
            return contexts
        return heapq.nlargest(topk, contexts, key=lambda c: c.score)

    def _rg_search(self, repo_root: Path, query: str) -> list[dict[str, Any]]:
        cmd = [
            "rg",
            "-n",
            "--no-heading",
            "--color",
            "never",
            query,
            ".",
        ]
        try:
            rr = self._runner.run(cmd, cwd=str(repo_root), timeout_s=self._config.limits.rg_timeout_s)
        except (TimeoutError, SubprocessError):
            return []

        if rr.exit_code not in (0, 1):
            return []
        out: list[dict[str, Any]] = []
        for line in rr.stdout.splitlines():
            parts = line.split(":", 2)
            if len(parts) != 3:
                continue
            path_s, line_s, text = parts
            try:
                ln = int(line_s)
            except ValueError:
                continue
            out.append({"path": path_s, "line": ln, "text": text})
            if len(out) >= 200:
                break
        return out


def _vector_hits_to_contexts(hits: list[Any], repo_root: Path) -> list[RetrievedContext]:
    """将 Chroma VectorHit 转为 RetrievedContext，保证 path/行号来自索引，减轻幻觉。"""
    out: list[RetrievedContext] = []
    for h in hits:
        if not isinstance(h, VectorHit):
            continue
        path = h.metadata.get("path")
        start_line = h.metadata.get("start_line")
        end_line = h.metadata.get("end_line")
        if path is None or start_line is None or end_line is None:
            continue
        try:
            sl, el = int(start_line), int(end_line)
        except (TypeError, ValueError):
            continue
        snippet = h.snippet if getattr(h, "snippet", None) else ""
        if not snippet and path:
            snippet = _read_lines(repo_root / path, sl, el)
        out.append(
            RetrievedContext(path=str(path), start_line=sl, end_line=el, snippet=snippet or "", score=h.score)
        )
    return out


def _merge_contexts(rg_ctx: list[RetrievedContext], chroma_ctx: list[RetrievedContext], topk: int) -> list[RetrievedContext]:
    """合并 rg 与 Chroma 结果，按 (path, start_line) 去重保留高分，再取 topk。"""
    seen: set[tuple[str, int]] = set()
    merged: list[RetrievedContext] = []
    for c in rg_ctx:
        key = (c.path, c.start_line)
        if key not in seen:
            seen.add(key)
            merged.append(c)
    for c in chroma_ctx:
        key = (c.path, c.start_line)
        if key not in seen:
            seen.add(key)
            merged.append(c)
    merged.sort(key=lambda x: x.score, reverse=True)
    return merged[:topk] if len(merged) > topk else merged


class HybridRetriever:
    """rg 关键词检索 + Chroma 语义检索，合并结果；Chroma 提供真实 path/行号，减轻路径幻觉。"""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._keyword = KeywordRetriever(config)
        self._chroma_dir = config.repo_root() / CHROMA_PERSIST_DIR
        self._vector_store: ChromaVectorStore | None = None

    def _get_store(self) -> ChromaVectorStore | None:
        if self._vector_store is not None:
            return self._vector_store
        if not self._chroma_dir.is_dir():
            return None
        try:
            self._vector_store = ChromaVectorStore(str(self._chroma_dir))
            return self._vector_store
        except Exception:
            return None

    def retrieve(self, repo_root: Path, query: str, *, topk: int | None = None) -> list[RetrievedContext]:
        topk = topk or self._config.context.retrieved_topk
        rg_ctx = self._keyword.retrieve(repo_root, query, topk=topk * 2)  # 多取一些便于与 Chroma 合并后截断
        store = self._get_store()
        if store is None:
            return rg_ctx[:topk] if len(rg_ctx) > topk else rg_ctx
        try:
            hits = store.query(query, topk=topk)
            chroma_ctx = _vector_hits_to_contexts(hits, repo_root)
        except Exception:
            chroma_ctx = []
        return _merge_contexts(rg_ctx, chroma_ctx, topk)


def _read_lines(path: Path, start_line: int, end_line: int) -> str:
    """Read only the requested line range to avoid loading large files into memory."""
    out_lines: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, start=1):
                if i > end_line:
                    break
                if i >= start_line:
                    out_lines.append(f"{i:6d}| {line.rstrip('\n\r')}")
    except OSError:
        return ""
    return "\n".join(out_lines)


def _merge_line_windows(lines: list[int], *, window: int, max_lines: int) -> list[tuple[int, int]]:
    if not lines:
        return []
    ranges: list[tuple[int, int]] = []
    cur_start = max(1, lines[0] - window)
    cur_end = lines[0] + window
    for ln in lines[1:]:
        s = max(1, ln - window)
        e = ln + window
        if s <= cur_end + 20:
            cur_end = max(cur_end, e)
        else:
            ranges.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    ranges.append((cur_start, cur_end))

    trimmed: list[tuple[int, int]] = []
    for s, e in ranges:
        if (e - s + 1) > max_lines:
            e = s + max_lines - 1
        trimmed.append((s, e))
    return trimmed

