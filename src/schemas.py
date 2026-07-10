"""Shared schemas for MemOS-lite.

This module intentionally stays small.  The MVP uses plaintext memory plus
metadata, not the full MemOS MemCube with activation tensors or parameter
patches.  The fields here are enough for Stage 1: persistent memory units,
minimal provenance, lifecycle, hot/cold tiering, duplicate/conflict records,
QA results, and answer cache records.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def now_utc() -> datetime:
    """Return a timezone-naive UTC timestamp.

    The whole MVP stores ISO strings in SQLite and compares naive datetimes.
    This keeps the code simple and consistent on a local laptop.  If the system
    later becomes multi-region, switch this function and all DB parsing to
    timezone-aware UTC datetimes in one pass.
    """

    return datetime.utcnow()


def new_id() -> str:
    """Create a stable, opaque id for one memory unit."""

    return str(uuid.uuid4())


def normalize_for_hash(text: str) -> str:
    """Normalize text before exact-duplicate hashing.

    This is deliberately conservative: it ignores whitespace/case differences,
    but does not try to rewrite semantics.  Near-duplicate and conflict checks
    should live in memory_ops.py / duplicate_conflict.py.
    """

    text = text or ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash used for exact duplicate detection."""

    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


def compute_query_hash(query: str) -> str:
    """Hash a normalized question for exact QA-cache lookup."""

    return hashlib.sha256(normalize_for_hash(query).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MemoryStatus(str, Enum):
    """Lifecycle status for a memory unit.

    Keep this as lifecycle state only.  Hot/cold serving priority is represented
    separately by MemoryTier so the code does not confuse "expired" with "cold".
    """

    ACTIVE = "active"
    ARCHIVED = "archived"
    EXPIRED = "expired"


class MemoryTier(str, Enum):
    """Serving tier used by the scheduler.

    HOT: frequently reused memory, e.g. stable FAQ.
    WARM: normal active memory.
    COLD: low-priority memory; normally still persisted but not preferred.
    """

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


class SourceType(str, Enum):
    """Minimal source types for Stage 1 provenance."""

    DOC = "doc"              # business .docx files
    FAQ = "faq"              # FAQ .xlsx rows
    USER_INPUT = "user_input"
    MANUAL = "manual"
    INFERRED = "inferred"    # reserved for later conversation extraction


class MemoryType(str, Enum):
    """Memory representation type.

    The MVP only stores PLAINTEXT.  ACTIVATION and PARAMETER are kept as labels
    so later code can report or experiment with KV/prefix cache without changing
    the public schema.
    """

    PLAINTEXT = "plaintext"
    ACTIVATION = "activation"
    PARAMETER = "parameter"


class ConflictType(str, Enum):
    """Conflict category stored in the conflicts table."""

    DUPLICATE = "duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    CONTRADICTION = "contradiction"


class ConflictResolution(str, Enum):
    """Admin resolution state for a conflict record."""

    UNRESOLVED = "unresolved"
    KEPT_A = "kept_a"
    KEPT_B = "kept_b"
    MERGED = "merged"
    IGNORED = "ignored"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Provenance:
    """Minimal provenance attached to a memory unit."""

    source_type: SourceType
    source_path: str = ""       # e.g. data/raw/vay_tien.docx
    source_ref: str = ""        # e.g. sheet name + row id, or doc section
    ingested_at: datetime = field(default_factory=now_utc)
    ingested_by: str = "system"


@dataclass
class MemoryUnit:
    """Smallest managed knowledge unit in MemOS-lite.

    Compared with plain RAG chunks, this object carries lifecycle metadata,
    provenance link, usage counters, TTL, duplicate hash, and a serving tier.
    """

    content: str
    source: str
    category: str = "uncategorized"
    tags: list[str] = field(default_factory=list)
    summary: str = ""

    id: str = field(default_factory=new_id)
    memory_type: MemoryType = MemoryType.PLAINTEXT
    status: MemoryStatus = MemoryStatus.ACTIVE
    tier: MemoryTier = MemoryTier.WARM

    created_at: datetime = field(default_factory=now_utc)
    updated_at: datetime = field(default_factory=now_utc)
    last_accessed_at: Optional[datetime] = None
    access_count: int = 0
    ttl_expires_at: Optional[datetime] = None

    content_hash: str = ""
    version: int = 1
    provenance: Optional[Provenance] = None

    # Vector is stored in Chroma/vector_store.py.  SQLite keeps only the id.
    embedding_id: Optional[str] = None

    # Escape hatch for small metadata without adding a column each time.
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.memory_type, str):
            self.memory_type = MemoryType(self.memory_type)
        if isinstance(self.status, str):
            self.status = MemoryStatus(self.status)
        if isinstance(self.tier, str):
            self.tier = MemoryTier(self.tier)
        if not self.content_hash:
            self.content_hash = compute_content_hash(self.content)

    @property
    def is_available(self) -> bool:
        """True when the unit can be used for QA retrieval."""

        return self.status not in {MemoryStatus.EXPIRED, MemoryStatus.ARCHIVED}

    def refresh_hash(self) -> None:
        """Recompute hash after content edits."""

        self.content_hash = compute_content_hash(self.content)
        self.updated_at = now_utc()


@dataclass
class RetrievedMemory:
    """One retrieval result returned by retriever/memory_manager."""

    memory: MemoryUnit
    score: float
    rank: int = 0


@dataclass
class ConflictRecord:
    """Conflict/duplicate record returned from db.py."""

    id: int
    memory_id_a: str
    memory_id_b: str
    conflict_type: ConflictType = ConflictType.DUPLICATE
    resolution: ConflictResolution = ConflictResolution.UNRESOLVED
    detected_at: datetime = field(default_factory=now_utc)
    resolved_at: Optional[datetime] = None
    note: str = ""


@dataclass
class QAResult:
    """Output of one QA call, shared by baseline and MemOS-lite evaluation."""

    query: str
    answer: str
    branch: str                      # "baseline" | "memos"
    latency_ms: float = 0.0
    retrieved_sources: list[str] = field(default_factory=list)
    retrieved_memory_ids: list[str] = field(default_factory=list)
    cache_hit: bool = False
    raw_context: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheEntry:
    """One QA cache row.

    retrieved_memory_ids lets qa_cache.py verify that cached answers still depend
    only on active/non-expired memory before reuse.
    """

    query_hash: str
    query: str
    answer: str
    created_at: datetime = field(default_factory=now_utc)
    expires_at: Optional[datetime] = None
    hit_count: int = 0
    retrieved_memory_ids: list[str] = field(default_factory=list)
