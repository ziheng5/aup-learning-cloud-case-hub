"""
Optional vector store integration (FAISS/Chroma).

This project keeps the interface stubbed so the architecture can be extended
without changing the Orchestrator contracts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VectorHit:
    id: str
    score: float
    metadata: dict
    snippet: str | None = None  # 检索时带出原文，便于转成 RetrievedContext


class VectorStore:
    def add(self, id: str, text: str, metadata: dict) -> None:  # pragma: no cover
        raise NotImplementedError

    def query(self, text: str, *, topk: int) -> list[VectorHit]:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Chroma 接入：约定数据目录 .jupypilot_chroma，用 sentence-transformers 做向量
# ---------------------------------------------------------------------------

CHROMA_PERSIST_DIR = ".jupypilot_chroma"
CHROMA_COLLECTION_NAME = "jupypilot"


def _get_chroma_client():
    import chromadb
    return chromadb.PersistentClient


def _get_embedding_function():
    from chromadb.utils import embedding_functions
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


class ChromaVectorStore(VectorStore):
    """Chroma 向量库实现，数据持久化到 persist_directory（如 .jupypilot_chroma）。"""

    def __init__(
        self,
        persist_directory: str,
        *,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ) -> None:
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._client = _get_chroma_client()(path=persist_directory)
        self._embedding_fn = _get_embedding_function()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"description": "JupyPilot code/doc chunks"},
        )

    def add(self, id: str, text: str, metadata: dict) -> None:
        # Chroma 要求 metadata 值多为 str/int/float；复杂结构需序列化
        safe_meta = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in metadata.items()}
        self._collection.add(ids=[id], documents=[text], metadatas=[safe_meta])

    def add_batch(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        """批量写入，建索引时更高效。"""
        if not ids:
            return
        safe_metas = [
            {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in m.items()}
            for m in metadatas
        ]
        self._collection.add(ids=ids, documents=texts, metadatas=safe_metas)

    def query(self, text: str, *, topk: int) -> list[VectorHit]:
        result = self._collection.query(
            query_texts=[text],
            n_results=topk,
            include=["documents", "metadatas", "distances"],
        )
        if not result["ids"] or not result["ids"][0]:
            return []
        hits: list[VectorHit] = []
        docs = (result.get("documents") or ([[]]))[0]
        for i, doc_id in enumerate(result["ids"][0]):
            dist = (result["distances"][0][i]) if result.get("distances") else 0.0
            meta = (result["metadatas"][0][i]) if result.get("metadatas") else {}
            snippet = docs[i] if i < len(docs) else None
            score = float(-dist) if dist is not None else 0.0
            hits.append(VectorHit(id=doc_id, score=score, metadata=meta, snippet=snippet))
        return hits

