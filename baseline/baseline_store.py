"""Vector store độc lập cho RAG baseline.

Baseline dùng Chroma collection riêng, không dùng chung src/vector_store.py, để hai
nhánh MemOS-lite và RAG thường tách biệt khi eval_compare.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

COLLECTION_NAME = "baseline_chunks"


class BaselineVectorStore:
    """Bọc một Chroma persistent collection riêng cho baseline."""

    def __init__(self, chroma_path: str | Path, collection_name: str = COLLECTION_NAME):
        self.chroma_path = Path(chroma_path)
        self.collection_name = collection_name
        self._client: Any = None
        self._collection: Any = None

    def _ensure_client(self) -> None:
        if self._collection is not None:
            return
        import chromadb

        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.chroma_path))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _clean_metadata(metadata: Optional[dict[str, Any]]) -> dict[str, str | int | float | bool]:
        """Chroma chỉ nhận metadata phẳng."""

        clean: dict[str, str | int | float | bool] = {}
        for key, value in (metadata or {}).items():
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

    @staticmethod
    def _distance_to_score(distance: Any) -> float:
        if distance is None:
            return 0.0
        try:
            # Với Chroma cosine distance: 0 là giống nhất, càng lớn càng xa.
            return max(0.0, min(1.0, 1.0 - float(distance)))
        except (TypeError, ValueError):
            return 0.0

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Upsert chunks vào baseline collection."""

        if not ids:
            return
        if not (len(ids) == len(embeddings) == len(documents)):
            raise ValueError("ids, embeddings, documents phải có cùng độ dài")
        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError("metadatas phải có cùng độ dài với ids")

        self._ensure_client()
        cleaned = [self._clean_metadata(m) for m in (metadatas or [{} for _ in ids])]
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=cleaned,
        )

    # Alias để code gọi upsert/add đều được.
    upsert = add

    def query(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        """Vector search thuần.

        Không có filter lifecycle/status/tier vì baseline không quản lý memory.
        """

        if top_k <= 0:
            return []
        self._ensure_client()
        if self.count() == 0:
            return []

        result = self._collection.query(query_embeddings=[query_embedding], n_results=top_k)
        hits: list[dict[str, Any]] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        for idx, doc_id in enumerate(ids):
            distance = dists[idx] if idx < len(dists) else None
            hits.append(
                {
                    "id": doc_id,
                    "document": docs[idx] if idx < len(docs) else "",
                    "metadata": metas[idx] if idx < len(metas) else {},
                    "distance": distance,
                    "score": self._distance_to_score(distance),
                }
            )
        return hits

    def count(self) -> int:
        self._ensure_client()
        return int(self._collection.count())

    def delete(self, ids: list[str]) -> None:
        """Xóa một số chunks theo id, dùng cho test/debug."""

        if not ids:
            return
        self._ensure_client()
        self._collection.delete(ids=ids)

    def reset(self) -> None:
        """Xóa toàn bộ baseline collection, dùng khi muốn ingest lại từ đầu."""

        self._ensure_client()
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
