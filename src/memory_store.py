"""Unified storage facade for MemOS-lite.

SQLite is the source of truth for MemoryUnit metadata. Chroma stores only
embeddings and lightweight filtering metadata. This class keeps the two stores
behind one small interface used by ingest, retrieval, memory_ops, and scheduler.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from datetime import timedelta
from typing import Any, Optional

from src import db
from src.config import Config
from src.schemas import (
    ConflictRecord,
    ConflictResolution,
    ConflictType,
    MemoryStatus,
    MemoryTier,
    MemoryUnit,
    RetrievedMemory,
    now_utc,
)
from src.vector_store import VectorStore

EmbedFn = Callable[[str], Sequence[float]]


class MemoryStore:
    """Facade over SQLite + Chroma.

    It intentionally does not decide duplicate/conflict policy. That logic lives
    in memory_ops.py. It only persists, updates, and retrieves memory units.
    """

    def __init__(self, config: Config, embed_fn: Optional[EmbedFn] = None) -> None:
        self.config = config
        self.embed_fn = embed_fn
        db.init_db(config.sqlite_path)
        self.vector_store = VectorStore(
            config.chroma_path,
            collection_name=config.memos_collection,
        )

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        if self.embed_fn is None:
            raise RuntimeError("MemoryStore requires embed_fn for vector operations")
        return [float(x) for x in self.embed_fn(text)]

    @staticmethod
    def _score_from_distance(distance: Any) -> float:
        if distance is None:
            return 0.0
        try:
            # Chroma cosine distance: lower is better, 0 means identical.
            return max(0.0, min(1.0, 1.0 - float(distance)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _vector_metadata(unit: MemoryUnit) -> dict[str, Any]:
        source_type = unit.provenance.source_type.value if unit.provenance else ""
        source_ref = unit.provenance.source_ref if unit.provenance else ""
        return {
            "source": unit.source,
            "source_type": source_type,
            "source_ref": source_ref,
            "category": unit.category,
            "tags": unit.tags,
            "status": unit.status.value,
            "tier": unit.tier.value,
            "memory_type": unit.memory_type.value,
            "version": unit.version,
        }

    def _apply_default_ttl(self, unit: MemoryUnit) -> None:
        if unit.ttl_expires_at is None and self.config.default_ttl_seconds > 0:
            unit.ttl_expires_at = unit.created_at + timedelta(seconds=self.config.default_ttl_seconds)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def add(self, unit: MemoryUnit, *, apply_default_ttl: bool = True) -> MemoryUnit:
        """Persist a new memory unit in SQLite and optionally Chroma."""

        if apply_default_ttl:
            self._apply_default_ttl(unit)

        if self.embed_fn is not None:
            unit.embedding_id = unit.id

        with db.connect(self.config.sqlite_path) as conn:
            db.insert_memory_unit(conn, unit)

        if self.embed_fn is not None:
            self.vector_store.upsert(
                ids=[unit.id],
                embeddings=[self._embed(unit.content)],
                documents=[unit.content],
                metadatas=[self._vector_metadata(unit)],
            )
        return unit

    def update(self, unit: MemoryUnit, *, reembed: bool = True) -> MemoryUnit:
        """Overwrite a MemoryUnit. Use update_content() for versioned content edits."""

        unit.updated_at = now_utc()
        if self.embed_fn is not None:
            unit.embedding_id = unit.id

        with db.connect(self.config.sqlite_path) as conn:
            db.update_memory_unit(conn, unit)

        if reembed and self.embed_fn is not None:
            self.vector_store.upsert(
                ids=[unit.id],
                embeddings=[self._embed(unit.content)],
                documents=[unit.content],
                metadatas=[self._vector_metadata(unit)],
            )
        elif self.embed_fn is not None:
            self.vector_store.update_metadata([unit.id], [self._vector_metadata(unit)])
        return unit

    def update_content(self, memory_id: str, new_content: str, *, summary: Optional[str] = None) -> MemoryUnit:
        """Versioned content update through db.update_memory_content()."""

        with db.connect(self.config.sqlite_path) as conn:
            unit = db.update_memory_content(conn, memory_id, new_content, summary=summary)

        if self.embed_fn is not None:
            unit.embedding_id = unit.id
            self.vector_store.upsert(
                ids=[unit.id],
                embeddings=[self._embed(unit.content)],
                documents=[unit.content],
                metadatas=[self._vector_metadata(unit)],
            )
            with db.connect(self.config.sqlite_path) as conn:
                db.update_memory_unit(conn, unit)
        return unit

    def delete(self, memory_id: str, *, hard: bool = False) -> None:
        """Archive by default; hard delete only when explicitly requested."""

        if hard:
            with db.connect(self.config.sqlite_path) as conn:
                db.delete_memory_unit(conn, memory_id)
            if self.embed_fn is not None:
                self.vector_store.delete([memory_id])
            return
        self.set_status(memory_id, MemoryStatus.ARCHIVED, reason="archive_by_delete_api")

    def set_status(self, memory_id: str, status: MemoryStatus | str, reason: str = "") -> None:
        with db.connect(self.config.sqlite_path) as conn:
            db.update_status(conn, memory_id, status, reason=reason)
            unit = db.get_memory_unit(conn, memory_id)

        if unit is not None and self.embed_fn is not None:
            self.vector_store.update_metadata([memory_id], [self._vector_metadata(unit)])

    def set_tier(self, memory_id: str, tier: MemoryTier | str, reason: str = "") -> None:
        with db.connect(self.config.sqlite_path) as conn:
            db.update_tier(conn, memory_id, tier, reason=reason)
            unit = db.get_memory_unit(conn, memory_id)

        if unit is not None and self.embed_fn is not None:
            self.vector_store.update_metadata([memory_id], [self._vector_metadata(unit)])

    def touch_access(self, memory_id: str) -> None:
        with db.connect(self.config.sqlite_path) as conn:
            db.touch_access(conn, memory_id, accessed_at=now_utc())

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get(self, memory_id: str) -> Optional[MemoryUnit]:
        with db.connect(self.config.sqlite_path) as conn:
            return db.get_memory_unit(conn, memory_id)

    def list(
        self,
        status: Optional[MemoryStatus | str] = None,
        category: Optional[str] = None,
        tier: Optional[MemoryTier | str] = None,
        include_expired: bool = True,
        limit: int = 100,
    ) -> list[MemoryUnit]:
        with db.connect(self.config.sqlite_path) as conn:
            return db.list_memory_units(
                conn,
                status=status,
                category=category,
                tier=tier,
                include_expired=include_expired,
                limit=limit,
            )

    def find_by_content_hash(self, content_hash: str) -> list[MemoryUnit]:
        with db.connect(self.config.sqlite_path) as conn:
            return db.find_by_content_hash(conn, content_hash)

    def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        status: Optional[MemoryStatus | str] = MemoryStatus.ACTIVE,
        category: Optional[str] = None,
        tier: Optional[MemoryTier | str] = None,
        min_score: Optional[float] = None,
        touch: bool = True,
    ) -> list[RetrievedMemory]:
        """Vector search, then hydrate/filter using SQLite as source of truth."""

        if self.embed_fn is None:
            return []

        where: dict[str, Any] = {}
        if status is not None:
            where["status"] = status.value if hasattr(status, "value") else str(status)
        if category:
            where["category"] = category
        if tier is not None:
            where["tier"] = tier.value if hasattr(tier, "value") else str(tier)

        hits = self.vector_store.query(
            self._embed(query),
            top_k=top_k or self.config.top_k,
            where=where or None,
        )
        threshold = self.config.min_retrieval_score if min_score is None else min_score

        results: list[RetrievedMemory] = []
        seen: set[str] = set()
        for hit in hits:
            memory_id = str(hit.get("id", ""))
            if not memory_id or memory_id in seen:
                continue
            seen.add(memory_id)

            unit = self.get(memory_id)
            if unit is None or not unit.is_available:
                continue
            if category and unit.category != category:
                continue
            if status is not None and unit.status.value != (status.value if hasattr(status, "value") else str(status)):
                continue
            if tier is not None and unit.tier.value != (tier.value if hasattr(tier, "value") else str(tier)):
                continue

            score = self._score_from_distance(hit.get("distance"))
            if score < threshold:
                continue
            if touch:
                self.touch_access(unit.id)
                refreshed = self.get(unit.id)
                if refreshed is not None:
                    unit = refreshed
            results.append(RetrievedMemory(memory=unit, score=score, rank=len(results) + 1))
        return results

    # ------------------------------------------------------------------
    # Conflict/lifecycle helpers used by memory_ops/scheduler
    # ------------------------------------------------------------------

    def add_conflict(
        self,
        memory_id_a: str,
        memory_id_b: str,
        conflict_type: ConflictType | str = ConflictType.NEAR_DUPLICATE,
        note: str = "",
    ) -> int:
        with db.connect(self.config.sqlite_path) as conn:
            return db.insert_conflict(conn, memory_id_a, memory_id_b, conflict_type, note=note)

    def list_conflicts(
        self,
        resolution: Optional[ConflictResolution | str] = None,
        limit: int = 100,
    ) -> list[ConflictRecord]:
        with db.connect(self.config.sqlite_path) as conn:
            return db.list_conflicts(conn, resolution=resolution, limit=limit)

    def expire_due(self) -> int:
        with db.connect(self.config.sqlite_path) as conn:
            return db.expire_due_memories(conn, now=now_utc())

    def reindex(self, units: Optional[Iterable[MemoryUnit]] = None) -> int:
        """Rebuild Chroma vectors from SQLite units when needed."""

        if self.embed_fn is None:
            return 0
        units_to_index = list(units) if units is not None else self.list(include_expired=False, limit=100_000)
        if not units_to_index:
            return 0
        self.vector_store.upsert(
            ids=[unit.id for unit in units_to_index],
            embeddings=[self._embed(unit.content) for unit in units_to_index],
            documents=[unit.content for unit in units_to_index],
            metadatas=[self._vector_metadata(unit) for unit in units_to_index],
        )
        return len(units_to_index)
