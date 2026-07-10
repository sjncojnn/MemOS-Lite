"""Small Chroma wrapper for MemOS-lite vector storage.

This module stores only vector-facing data: id, document text, and a few flat
metadata fields used for fast filtering. All business state (lifecycle,
provenance, conflicts, versions, cache) stays in SQLite/db.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

DEFAULT_COLLECTION_NAME = "memos_lite_memories"


class VectorStore:
    """Persistent Chroma collection wrapper.

    Chroma is imported lazily so tests that only touch SQLite do not need to
    initialize the vector backend.
    """

    def __init__(
        self,
        chroma_path: str | Path,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.chroma_path = Path(chroma_path)
        self.collection_name = collection_name
        self._client: Any = None
        self._collection: Any = None

    def _ensure_client(self) -> None:
        if self._collection is not None:
            return

        import chromadb  # imported only when vector storage is actually used

        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.chroma_path))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _build_where(where: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """Convert simple filters to Chroma-compatible where syntax.

        Chroma accepts:
            {"status": "active"}

        But when there are multiple fields, newer versions require:
            {"$and": [{"status": "active"}, {"category": "faq"}]}
        """

        clean = VectorStore._clean_metadata(where or {})
        if not clean:
            return None

        clauses = [{key: value} for key, value in clean.items()]
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @staticmethod
    def _clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
        """Keep metadata flat because Chroma does not accept nested/list values."""

        clean: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif hasattr(value, "value"):
                clean[key] = str(value.value)
            elif isinstance(value, (list, tuple, set)):
                clean[key] = ",".join(str(item) for item in value if item is not None)
            else:
                clean[key] = str(value)
        return clean

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Insert or overwrite vectors by id."""

        if not ids:
            return
        self._ensure_client()
        cleaned = [self._clean_metadata(m) for m in (metadatas or [{} for _ in ids])]
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=cleaned,
        )

    # Backward-compatible alias for the skeleton API.
    add = upsert

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Return Chroma hits as {id, document, metadata, distance}."""

        self._ensure_client()
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, top_k),
            where=self._build_where(where),
        )

        hits: list[dict[str, Any]] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        for idx, memory_id in enumerate(ids):
            hits.append(
                {
                    "id": memory_id,
                    "document": docs[idx] if idx < len(docs) else "",
                    "metadata": metas[idx] if idx < len(metas) else {},
                    "distance": dists[idx] if idx < len(dists) else None,
                }
            )
        return hits

    def update_metadata(self, ids: list[str], metadatas: list[dict[str, Any]]) -> None:
        """Update only vector metadata, useful after status/tier changes."""

        if not ids:
            return
        self._ensure_client()
        cleaned = [self._clean_metadata(m) for m in metadatas]
        self._collection.update(ids=ids, metadatas=cleaned)

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self._ensure_client()
        self._collection.delete(ids=ids)

    def count(self) -> int:
        self._ensure_client()
        return int(self._collection.count())
