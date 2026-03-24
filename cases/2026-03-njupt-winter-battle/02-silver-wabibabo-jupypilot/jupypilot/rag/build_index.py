"""
建索引脚本：扫描项目代码/文档，分片后写入 Chroma（.jupypilot_chroma）。

用法（在项目根目录执行）：
  python -m jupypilot.rag.build_index
或指定 repo 路径：
  python -m jupypilot.rag.build_index --repo-path "D:/path/to/JupyPilot"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_config
from .scanner import RepoScanner
from .vector_store import CHROMA_PERSIST_DIR, ChromaVectorStore

# 分片参数：与检索时的上下文窗口协调；更小 stride 产生更多 chunk，覆盖更细
CHUNK_LINES = 150
CHUNK_STRIDE = 100  # 相邻块重叠 50 行，提高召回、减少漏段
BATCH_SIZE = 32  # 每批写入 Chroma 的 chunk 数


def _chunk_file(content: str, path: str) -> list[tuple[str, str, dict]]:
    """按行分片，返回 (id, text, metadata) 列表。"""
    lines = content.splitlines()
    if not lines:
        return []
    chunks: list[tuple[str, str, dict]] = []
    start = 1
    while start <= len(lines):
        end = min(start + CHUNK_LINES - 1, len(lines))
        block = "\n".join(lines[start - 1 : end])
        chunk_id = f"{path}:{start}-{end}"
        meta = {"path": path, "start_line": start, "end_line": end}
        chunks.append((chunk_id, block, meta))
        if end >= len(lines):
            break
        start += CHUNK_STRIDE
    return chunks


def build_index(repo_root: Path, *, persist_dir: str | None = None) -> tuple[int, int, int, int]:
    """
    对 repo_root 下符合 RagConfig 的文件建 Chroma 索引。
    返回 (total_chunks, files_scanned, files_indexed, files_skipped)。
    """
    config = load_config()
    rag_config = config.rag
    repo_root = repo_root.resolve()
    persist_directory = str(repo_root / (persist_dir or CHROMA_PERSIST_DIR))

    scanner = RepoScanner(rag_config)
    files = scanner.scan(repo_root)
    store = ChromaVectorStore(persist_directory)

    total = 0
    files_indexed = 0
    files_skipped = 0
    ids_buf: list[str] = []
    texts_buf: list[str] = []
    metas_buf: list[dict] = []

    for sf in files:
        full_path = repo_root / sf.path
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            files_skipped += 1
            continue
        chunks_for_file = _chunk_file(content, sf.path)
        if not chunks_for_file:
            continue
        files_indexed += 1
        for chunk_id, text, meta in chunks_for_file:
            ids_buf.append(chunk_id)
            texts_buf.append(text)
            metas_buf.append(meta)
            if len(ids_buf) >= BATCH_SIZE:
                store.add_batch(ids_buf, texts_buf, metas_buf)
                total += len(ids_buf)
                ids_buf, texts_buf, metas_buf = [], [], []

    if ids_buf:
        store.add_batch(ids_buf, texts_buf, metas_buf)
        total += len(ids_buf)

    return total, len(files), files_indexed, files_skipped


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build Chroma index for JupyPilot repo.")
    parser.add_argument(
        "--repo-path",
        type=str,
        default=".",
        help="Repo root path (default: current directory)",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=CHROMA_PERSIST_DIR,
        help=f"Chroma persist directory name under repo (default: {CHROMA_PERSIST_DIR})",
    )
    args = parser.parse_args()
    repo_root = Path(args.repo_path).expanduser().resolve()
    if not repo_root.is_dir():
        raise SystemExit(f"Not a directory: {repo_root}")
    total, scanned, indexed, skipped = build_index(repo_root, persist_dir=args.persist_dir)
    print(f"Scanned {scanned} files, indexed {indexed} ({skipped} read errors). Total {total} chunks -> {repo_root / args.persist_dir}")


if __name__ == "__main__":
    _main()
