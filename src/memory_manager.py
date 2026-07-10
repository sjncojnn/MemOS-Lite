"""Central facade for MemOS-lite.

External scripts should use this class instead of calling db/vector/retriever/QA
modules directly. It keeps the public API MiniRAG-like: add, find, update,
answer, and run_scheduler.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Optional

from src import ingest, memory_ops
from src.client_factory import get_llm_client
from src.config import Config, load_config
from src.memory_store import MemoryStore
from src.qa_cache import QACache
from src.qa_service import QAService
from src.retriever import MemoryRetriever
from src.schemas import MemoryStatus, MemoryTier, MemoryUnit, QAResult, RetrievedMemory, now_utc
from src.scheduler import MemoryScheduler, SchedulerReport


class MemoryManager:
    """One clean entry point for Stage-1 MemOS-lite."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or load_config()
        self.llm_client = get_llm_client(self.config)
        self.store = MemoryStore(self.config, embed_fn=self.llm_client.embed)
        self.retriever = MemoryRetriever(self.store, self.config, embed_fn=self.llm_client.embed)
        self.cache = QACache(self.config, embed_fn=self.llm_client.embed)
        self.qa_service = QAService(self.retriever, self.llm_client, self.cache, self.config)
        self.scheduler = MemoryScheduler(self.store, self.config, cache=self.cache)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def add(
        self,
        unit: MemoryUnit,
        *,
        ttl_seconds: Optional[int] = None,
        skip_exact_duplicate: bool = True,
        record_near_duplicate: bool = True,
    ) -> Optional[MemoryUnit]:
        """Add one MemoryUnit. Return None when an exact duplicate is skipped."""

        if ttl_seconds is not None and ttl_seconds > 0:
            unit.ttl_expires_at = now_utc() + timedelta(seconds=ttl_seconds)
        stored, _ = memory_ops.add_memory(
            self.store,
            unit,
            skip_exact_duplicate=skip_exact_duplicate,
            record_near_duplicate=record_near_duplicate,
        )
        return stored

    def add_batch(self, units: list[MemoryUnit], *, ttl_seconds: Optional[int] = None) -> list[MemoryUnit]:
        """Add many units and return only newly inserted units."""

        stored: list[MemoryUnit] = []
        for unit in units:
            added = self.add(unit, ttl_seconds=ttl_seconds)
            if added is not None:
                stored.append(added)
        return stored

    def ingest_path(self, path: str | Path, *, category: str = "uncategorized") -> list[MemoryUnit]:
        """Load .docx/.xlsx from a file or directory and store as MemoryUnits."""

        units = ingest.load_path_as_units(path, config=self.config, category=category)
        return self.add_batch(units)

    def update(self, memory_id: str, **fields) -> Optional[MemoryUnit]:
        """Update a MemoryUnit. Content edits are versioned and re-indexed."""

        unit = self.store.get(memory_id)
        if unit is None:
            return None

        content = fields.pop("content", None)
        if content is not None and content != unit.content:
            summary = fields.pop("summary", None)
            unit = self.store.update_content(memory_id, str(content), summary=summary)

        changed = False
        for key, value in fields.items():
            if hasattr(unit, key):
                setattr(unit, key, value)
                changed = True
        if changed:
            unit = self.store.update(unit, reembed=False)
        return unit

    def delete(self, memory_id: str, *, hard: bool = False) -> None:
        """Archive by default; hard delete only when explicitly requested."""

        self.store.delete(memory_id, hard=hard)

    # ------------------------------------------------------------------
    # Read / QA path
    # ------------------------------------------------------------------

    def get(self, memory_id: str) -> Optional[MemoryUnit]:
        return self.store.get(memory_id)

    def list(
        self,
        *,
        status: Optional[MemoryStatus | str] = None,
        category: Optional[str] = None,
        tier: Optional[MemoryTier | str] = None,
        limit: int = 100,
    ) -> list[MemoryUnit]:
        return self.store.list(status=status, category=category, tier=tier, include_expired=False, limit=limit)

    def find(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        category: Optional[str] = None,
    ) -> list[RetrievedMemory]:
        return self.retriever.retrieve(query, top_k=top_k, category=category)

    def answer(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        category: Optional[str] = None,
    ) -> QAResult:
        return self.qa_service.answer(query, top_k=top_k, category=category)

    # ------------------------------------------------------------------
    # Lifecycle / maintenance
    # ------------------------------------------------------------------

    def run_scheduler(self) -> SchedulerReport:
        return self.scheduler.run()

    def set_status(self, memory_id: str, status: MemoryStatus, reason: str = "manual_status_update") -> None:
        self.store.set_status(memory_id, status, reason=reason)

    def set_tier(self, memory_id: str, tier: MemoryTier, reason: str = "manual_tier_update") -> None:
        self.store.set_tier(memory_id, tier, reason=reason)

    def health_check(self) -> bool:
        return self.llm_client.health_check()
