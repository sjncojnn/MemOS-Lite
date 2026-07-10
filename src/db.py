"""SQLite storage layer for MemOS-lite.

SQLite stores metadata and operational state only.  Embeddings/vectors should be
stored in Chroma through vector_store.py and referenced by embedding_id.

Tables:
- memory_units: plaintext memory payload + metadata/lifecycle/tier/version/hash
- provenance: minimal source tracking
- lifecycle_log: auditable state/tier/update events
- conflicts: exact/near duplicate and contradiction records
- qa_cache: stable answer cache metadata

No ORM, no framework, just sqlite3 and small helper functions.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from src.schemas import (
    CacheEntry,
    ConflictRecord,
    ConflictResolution,
    ConflictType,
    MemoryStatus,
    MemoryTier,
    MemoryType,
    MemoryUnit,
    Provenance,
    SourceType,
    compute_content_hash,
    now_utc,
)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memory_units (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'uncategorized',
    tags TEXT NOT NULL DEFAULT '[]',
    memory_type TEXT NOT NULL DEFAULT 'plaintext',
    status TEXT NOT NULL DEFAULT 'active',
    tier TEXT NOT NULL DEFAULT 'warm',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed_at TEXT,
    access_count INTEGER NOT NULL DEFAULT 0,
    ttl_expires_at TEXT,
    content_hash TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    embedding_id TEXT,
    extra_metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_memory_units_status ON memory_units(status);
CREATE INDEX IF NOT EXISTS idx_memory_units_tier ON memory_units(tier);
CREATE INDEX IF NOT EXISTS idx_memory_units_category ON memory_units(category);
CREATE INDEX IF NOT EXISTS idx_memory_units_content_hash ON memory_units(content_hash);
CREATE INDEX IF NOT EXISTS idx_memory_units_updated_at ON memory_units(updated_at);

CREATE TABLE IF NOT EXISTS provenance (
    memory_id TEXT PRIMARY KEY REFERENCES memory_units(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,
    source_path TEXT NOT NULL DEFAULT '',
    source_ref TEXT NOT NULL DEFAULT '',
    ingested_at TEXT NOT NULL,
    ingested_by TEXT NOT NULL DEFAULT 'system'
);

CREATE TABLE IF NOT EXISTS lifecycle_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL REFERENCES memory_units(id) ON DELETE CASCADE,
    event TEXT NOT NULL,
    from_status TEXT,
    to_status TEXT,
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_lifecycle_log_memory_id ON lifecycle_log(memory_id);

CREATE TABLE IF NOT EXISTS conflicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id_a TEXT NOT NULL REFERENCES memory_units(id) ON DELETE CASCADE,
    memory_id_b TEXT NOT NULL REFERENCES memory_units(id) ON DELETE CASCADE,
    conflict_type TEXT NOT NULL DEFAULT 'duplicate',
    resolution TEXT NOT NULL DEFAULT 'unresolved',
    detected_at TEXT NOT NULL,
    resolved_at TEXT,
    note TEXT NOT NULL DEFAULT '',
    UNIQUE(memory_id_a, memory_id_b, conflict_type)
);

CREATE INDEX IF NOT EXISTS idx_conflicts_resolution ON conflicts(resolution);
CREATE INDEX IF NOT EXISTS idx_conflicts_pair ON conflicts(memory_id_a, memory_id_b);

CREATE TABLE IF NOT EXISTS qa_cache (
    query_hash TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT,
    hit_count INTEGER NOT NULL DEFAULT 0,
    retrieved_memory_ids TEXT NOT NULL DEFAULT '[]'
);
"""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


def get_connection(sqlite_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with practical defaults for local MVP use."""

    path = Path(sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def init_db(sqlite_path: str | Path) -> None:
    """Create all tables/indexes.  Safe to call repeatedly on a fresh schema."""

    with get_connection(sqlite_path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


@contextmanager
def connect(sqlite_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Context manager that commits on success and rolls back on error."""

    conn = get_connection(sqlite_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _dt(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(value) if value else None


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_loads(value: Optional[str], default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _enum_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def _ordered_pair(memory_id_a: str, memory_id_b: str) -> tuple[str, str]:
    """Normalize conflict pairs so (A, B) and (B, A) are the same record."""

    if memory_id_a == memory_id_b:
        raise ValueError("Cannot create a conflict record for the same memory id")
    return tuple(sorted((memory_id_a, memory_id_b)))  # type: ignore[return-value]


def _row_to_unit(row: sqlite3.Row, provenance: Optional[Provenance] = None) -> MemoryUnit:
    return MemoryUnit(
        id=row["id"],
        content=row["content"],
        summary=row["summary"],
        source=row["source"],
        category=row["category"],
        tags=_json_loads(row["tags"], []),
        memory_type=MemoryType(row["memory_type"]),
        status=MemoryStatus(row["status"]),
        tier=MemoryTier(row["tier"]),
        created_at=_parse_dt(row["created_at"]) or now_utc(),
        updated_at=_parse_dt(row["updated_at"]) or now_utc(),
        last_accessed_at=_parse_dt(row["last_accessed_at"]),
        access_count=int(row["access_count"]),
        ttl_expires_at=_parse_dt(row["ttl_expires_at"]),
        content_hash=row["content_hash"],
        version=int(row["version"]),
        provenance=provenance,
        embedding_id=row["embedding_id"],
        extra_metadata=_json_loads(row["extra_metadata"], {}),
    )


def _row_to_conflict(row: sqlite3.Row) -> ConflictRecord:
    return ConflictRecord(
        id=int(row["id"]),
        memory_id_a=row["memory_id_a"],
        memory_id_b=row["memory_id_b"],
        conflict_type=ConflictType(row["conflict_type"]),
        resolution=ConflictResolution(row["resolution"]),
        detected_at=_parse_dt(row["detected_at"]) or now_utc(),
        resolved_at=_parse_dt(row["resolved_at"]),
        note=row["note"],
    )


def _row_to_cache(row: sqlite3.Row) -> CacheEntry:
    return CacheEntry(
        query_hash=row["query_hash"],
        query=row["query"],
        answer=row["answer"],
        created_at=_parse_dt(row["created_at"]) or now_utc(),
        expires_at=_parse_dt(row["expires_at"]),
        hit_count=int(row["hit_count"]),
        retrieved_memory_ids=_json_loads(row["retrieved_memory_ids"], []),
    )


# ---------------------------------------------------------------------------
# memory_units CRUD
# ---------------------------------------------------------------------------


def insert_memory_unit(conn: sqlite3.Connection, unit: MemoryUnit) -> None:
    """Insert one memory unit, optional provenance, and a lifecycle log row."""

    unit.refresh_hash()
    if not unit.created_at:
        unit.created_at = now_utc()
    unit.updated_at = unit.updated_at or unit.created_at

    conn.execute(
        """
        INSERT INTO memory_units
            (id, content, summary, source, category, tags, memory_type, status, tier,
             created_at, updated_at, last_accessed_at, access_count, ttl_expires_at,
             content_hash, version, embedding_id, extra_metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            unit.id,
            unit.content,
            unit.summary,
            unit.source,
            unit.category,
            _json_dumps(unit.tags),
            unit.memory_type.value,
            unit.status.value,
            unit.tier.value,
            _dt(unit.created_at),
            _dt(unit.updated_at),
            _dt(unit.last_accessed_at),
            unit.access_count,
            _dt(unit.ttl_expires_at),
            unit.content_hash,
            unit.version,
            unit.embedding_id,
            _json_dumps(unit.extra_metadata),
        ),
    )

    if unit.provenance:
        insert_provenance(conn, unit.id, unit.provenance)

    log_lifecycle_event(
        conn,
        unit.id,
        event="created",
        from_status=None,
        to_status=unit.status.value,
        reason="insert_memory_unit",
    )


def get_memory_unit(
    conn: sqlite3.Connection,
    memory_id: str,
    *,
    include_provenance: bool = True,
) -> Optional[MemoryUnit]:
    row = conn.execute("SELECT * FROM memory_units WHERE id = ?", (memory_id,)).fetchone()
    if row is None:
        return None
    provenance = get_provenance(conn, memory_id) if include_provenance else None
    return _row_to_unit(row, provenance=provenance)


def _attach_provenance(conn: sqlite3.Connection, units: list[MemoryUnit]) -> list[MemoryUnit]:
    """Batch-fetch provenance rows for a list of units and attach them in place.

    get_memory_unit() always attaches provenance for a single unit. Multi-row
    readers (list_memory_units, find_by_content_hash) must do the same so that
    provenance-dependent logic (e.g. memory_ops.is_hot(), FAQ filtering) behaves
    consistently no matter which read path was used.
    """

    if not units:
        return units
    ids = [u.id for u in units]
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT * FROM provenance WHERE memory_id IN ({placeholders})",
        ids,
    ).fetchall()
    provenance_by_id = {
        row["memory_id"]: Provenance(
            source_type=SourceType(row["source_type"]),
            source_path=row["source_path"],
            source_ref=row["source_ref"],
            ingested_at=_parse_dt(row["ingested_at"]) or now_utc(),
            ingested_by=row["ingested_by"],
        )
        for row in rows
    }
    for unit in units:
        unit.provenance = provenance_by_id.get(unit.id)
    return units


def find_by_content_hash(conn: sqlite3.Connection, content_hash: str) -> list[MemoryUnit]:
    """Find exact duplicates by content hash."""

    rows = conn.execute(
        "SELECT * FROM memory_units WHERE content_hash = ? ORDER BY created_at ASC",
        (content_hash,),
    ).fetchall()
    return _attach_provenance(conn, [_row_to_unit(row) for row in rows])


def list_memory_units(
    conn: sqlite3.Connection,
    status: Optional[MemoryStatus | str] = None,
    category: Optional[str] = None,
    tier: Optional[MemoryTier | str] = None,
    include_expired: bool = True,
    limit: int = 100,
) -> list[MemoryUnit]:
    """List memory units with small structured filters."""

    query = "SELECT * FROM memory_units WHERE 1=1"
    params: list[Any] = []

    if status is not None:
        query += " AND status = ?"
        params.append(_enum_value(status))
    elif not include_expired:
        query += " AND status NOT IN (?, ?)"
        params.extend([MemoryStatus.EXPIRED.value, MemoryStatus.ARCHIVED.value])

    if category:
        query += " AND category = ?"
        params.append(category)

    if tier is not None:
        query += " AND tier = ?"
        params.append(_enum_value(tier))

    query += " ORDER BY updated_at DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    return _attach_provenance(conn, [_row_to_unit(row) for row in rows])


def update_memory_unit(conn: sqlite3.Connection, unit: MemoryUnit) -> None:
    """Overwrite one memory unit by id.

    This function keeps the version passed in unit.  Higher-level update APIs can
    decide whether to increment version.
    """

    unit.refresh_hash()
    cur = conn.execute(
        """
        UPDATE memory_units SET
            content = ?, summary = ?, source = ?, category = ?, tags = ?,
            memory_type = ?, status = ?, tier = ?, updated_at = ?,
            last_accessed_at = ?, access_count = ?, ttl_expires_at = ?,
            content_hash = ?, version = ?, embedding_id = ?, extra_metadata = ?
        WHERE id = ?
        """,
        (
            unit.content,
            unit.summary,
            unit.source,
            unit.category,
            _json_dumps(unit.tags),
            unit.memory_type.value,
            unit.status.value,
            unit.tier.value,
            _dt(unit.updated_at),
            _dt(unit.last_accessed_at),
            unit.access_count,
            _dt(unit.ttl_expires_at),
            unit.content_hash,
            unit.version,
            unit.embedding_id,
            _json_dumps(unit.extra_metadata),
            unit.id,
        ),
    )
    if cur.rowcount == 0:
        raise ValueError(f"MemoryUnit not found: {unit.id}")

    if unit.provenance:
        insert_provenance(conn, unit.id, unit.provenance)


def update_memory_content(
    conn: sqlite3.Connection,
    memory_id: str,
    new_content: str,
    *,
    summary: Optional[str] = None,
    reason: str = "manual_update",
) -> MemoryUnit:
    """Update content with version increment and lifecycle log."""

    unit = get_memory_unit(conn, memory_id)
    if unit is None:
        raise ValueError(f"MemoryUnit not found: {memory_id}")

    old_status = unit.status.value
    unit.content = new_content
    if summary is not None:
        unit.summary = summary
    unit.version += 1
    unit.updated_at = now_utc()
    unit.content_hash = compute_content_hash(unit.content)

    update_memory_unit(conn, unit)
    log_lifecycle_event(
        conn,
        memory_id,
        event="updated",
        from_status=old_status,
        to_status=unit.status.value,
        reason=reason,
    )
    return unit


def touch_access(
    conn: sqlite3.Connection,
    memory_id: str,
    accessed_at: Optional[datetime] = None,
) -> None:
    """Update access_count and last_accessed_at after a successful retrieval."""

    accessed_at = accessed_at or now_utc()
    cur = conn.execute(
        """
        UPDATE memory_units
        SET access_count = access_count + 1,
            last_accessed_at = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (_dt(accessed_at), _dt(accessed_at), memory_id),
    )
    if cur.rowcount == 0:
        raise ValueError(f"MemoryUnit not found: {memory_id}")


def update_status(
    conn: sqlite3.Connection,
    memory_id: str,
    new_status: MemoryStatus | str,
    reason: str = "",
) -> None:
    """Change lifecycle status and write a log row."""

    new_status_value = _enum_value(new_status)
    row = conn.execute("SELECT status FROM memory_units WHERE id = ?", (memory_id,)).fetchone()
    if row is None:
        raise ValueError(f"MemoryUnit not found: {memory_id}")

    old_status = row["status"]
    if old_status == new_status_value:
        return

    conn.execute(
        "UPDATE memory_units SET status = ?, updated_at = ? WHERE id = ?",
        (new_status_value, _dt(now_utc()), memory_id),
    )
    log_lifecycle_event(
        conn,
        memory_id,
        event="status_change",
        from_status=old_status,
        to_status=new_status_value,
        reason=reason,
    )


def update_tier(
    conn: sqlite3.Connection,
    memory_id: str,
    new_tier: MemoryTier | str,
    reason: str = "",
) -> None:
    """Change hot/warm/cold serving tier and write a log row."""

    new_tier_value = _enum_value(new_tier)
    row = conn.execute("SELECT tier, status FROM memory_units WHERE id = ?", (memory_id,)).fetchone()
    if row is None:
        raise ValueError(f"MemoryUnit not found: {memory_id}")

    old_tier = row["tier"]
    if old_tier == new_tier_value:
        return

    conn.execute(
        "UPDATE memory_units SET tier = ?, updated_at = ? WHERE id = ?",
        (new_tier_value, _dt(now_utc()), memory_id),
    )
    log_lifecycle_event(
        conn,
        memory_id,
        event="tier_change",
        from_status=old_tier,
        to_status=new_tier_value,
        reason=reason,
    )


def expire_due_memories(conn: sqlite3.Connection, now: Optional[datetime] = None) -> int:
    """Mark memories as expired when ttl_expires_at has passed."""

    now = now or now_utc()
    rows = conn.execute(
        """
        SELECT id FROM memory_units
        WHERE ttl_expires_at IS NOT NULL
          AND ttl_expires_at <= ?
          AND status != ?
        """,
        (_dt(now), MemoryStatus.EXPIRED.value),
    ).fetchall()

    for row in rows:
        update_status(conn, row["id"], MemoryStatus.EXPIRED, reason="ttl_expired")

    return len(rows)


def delete_memory_unit(conn: sqlite3.Connection, memory_id: str) -> None:
    """Hard delete one unit.  Prefer ARCHIVED status in normal app flows."""

    conn.execute("DELETE FROM memory_units WHERE id = ?", (memory_id,))


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def insert_provenance(conn: sqlite3.Connection, memory_id: str, provenance: Provenance) -> None:
    conn.execute(
        """
        INSERT INTO provenance
            (memory_id, source_type, source_path, source_ref, ingested_at, ingested_by)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(memory_id) DO UPDATE SET
            source_type = excluded.source_type,
            source_path = excluded.source_path,
            source_ref = excluded.source_ref,
            ingested_at = excluded.ingested_at,
            ingested_by = excluded.ingested_by
        """,
        (
            memory_id,
            provenance.source_type.value,
            provenance.source_path,
            provenance.source_ref,
            _dt(provenance.ingested_at),
            provenance.ingested_by,
        ),
    )


def get_provenance(conn: sqlite3.Connection, memory_id: str) -> Optional[Provenance]:
    row = conn.execute("SELECT * FROM provenance WHERE memory_id = ?", (memory_id,)).fetchone()
    if row is None:
        return None
    return Provenance(
        source_type=SourceType(row["source_type"]),
        source_path=row["source_path"],
        source_ref=row["source_ref"],
        ingested_at=_parse_dt(row["ingested_at"]) or now_utc(),
        ingested_by=row["ingested_by"],
    )


# ---------------------------------------------------------------------------
# lifecycle_log
# ---------------------------------------------------------------------------


def log_lifecycle_event(
    conn: sqlite3.Connection,
    memory_id: str,
    event: str,
    from_status: Optional[str] = None,
    to_status: Optional[str] = None,
    reason: str = "",
) -> None:
    conn.execute(
        """
        INSERT INTO lifecycle_log
            (memory_id, event, from_status, to_status, reason, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (memory_id, event, from_status, to_status, reason, _dt(now_utc())),
    )


def get_lifecycle_log(conn: sqlite3.Connection, memory_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM lifecycle_log WHERE memory_id = ? ORDER BY id ASC",
        (memory_id,),
    ).fetchall()
    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# conflicts
# ---------------------------------------------------------------------------


def insert_conflict(
    conn: sqlite3.Connection,
    memory_id_a: str,
    memory_id_b: str,
    conflict_type: ConflictType | str = ConflictType.DUPLICATE,
    note: str = "",
) -> int:
    """Insert conflict if absent and return its id."""

    a, b = _ordered_pair(memory_id_a, memory_id_b)
    conflict_type_value = _enum_value(conflict_type)
    now = _dt(now_utc())

    conn.execute(
        """
        INSERT OR IGNORE INTO conflicts
            (memory_id_a, memory_id_b, conflict_type, resolution, detected_at, note)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (a, b, conflict_type_value, ConflictResolution.UNRESOLVED.value, now, note),
    )
    row = conn.execute(
        """
        SELECT id FROM conflicts
        WHERE memory_id_a = ? AND memory_id_b = ? AND conflict_type = ?
        """,
        (a, b, conflict_type_value),
    ).fetchone()
    if row is None:
        raise RuntimeError("Failed to insert or fetch conflict record")
    return int(row["id"])


def resolve_conflict(
    conn: sqlite3.Connection,
    conflict_id: int,
    resolution: ConflictResolution | str,
    note: Optional[str] = None,
) -> None:
    """Mark a conflict as resolved."""

    resolution_value = _enum_value(resolution)
    if note is None:
        conn.execute(
            "UPDATE conflicts SET resolution = ?, resolved_at = ? WHERE id = ?",
            (resolution_value, _dt(now_utc()), conflict_id),
        )
    else:
        conn.execute(
            "UPDATE conflicts SET resolution = ?, resolved_at = ?, note = ? WHERE id = ?",
            (resolution_value, _dt(now_utc()), note, conflict_id),
        )


def list_conflicts(
    conn: sqlite3.Connection,
    resolution: Optional[ConflictResolution | str] = None,
    limit: int = 100,
) -> list[ConflictRecord]:
    query = "SELECT * FROM conflicts WHERE 1=1"
    params: list[Any] = []
    if resolution is not None:
        query += " AND resolution = ?"
        params.append(_enum_value(resolution))
    query += " ORDER BY detected_at ASC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    return [_row_to_conflict(row) for row in rows]


def list_unresolved_conflicts(conn: sqlite3.Connection, limit: int = 100) -> list[ConflictRecord]:
    return list_conflicts(conn, resolution=ConflictResolution.UNRESOLVED, limit=limit)


# ---------------------------------------------------------------------------
# qa_cache
# ---------------------------------------------------------------------------


def upsert_qa_cache(
    conn: sqlite3.Connection,
    query_hash: str,
    query: str,
    answer: str,
    expires_at: Optional[datetime] = None,
    retrieved_memory_ids: Optional[Sequence[str]] = None,
) -> None:
    """Insert/update a QA cache entry.

    On update, hit_count is preserved.  qa_cache.py can implement semantic cache
    lookup on top of this exact-hash table if needed.
    """

    retrieved_memory_ids = list(retrieved_memory_ids or [])
    conn.execute(
        """
        INSERT INTO qa_cache
            (query_hash, query, answer, created_at, expires_at, hit_count, retrieved_memory_ids)
        VALUES (?, ?, ?, ?, ?, 0, ?)
        ON CONFLICT(query_hash) DO UPDATE SET
            query = excluded.query,
            answer = excluded.answer,
            expires_at = excluded.expires_at,
            retrieved_memory_ids = excluded.retrieved_memory_ids
        """,
        (
            query_hash,
            query,
            answer,
            _dt(now_utc()),
            _dt(expires_at),
            _json_dumps(retrieved_memory_ids),
        ),
    )


def get_qa_cache(conn: sqlite3.Connection, query_hash: str) -> Optional[CacheEntry]:
    row = conn.execute("SELECT * FROM qa_cache WHERE query_hash = ?", (query_hash,)).fetchone()
    return _row_to_cache(row) if row else None


def get_valid_qa_cache(
    conn: sqlite3.Connection,
    query_hash: str,
    now: Optional[datetime] = None,
    *,
    increment_hit: bool = True,
) -> Optional[CacheEntry]:
    """Return cache only if it has not expired."""

    now = now or now_utc()
    entry = get_qa_cache(conn, query_hash)
    if entry is None:
        return None
    if entry.expires_at is not None and entry.expires_at <= now:
        return None
    if increment_hit:
        increment_qa_cache_hit(conn, query_hash)
        entry.hit_count += 1
    return entry


def increment_qa_cache_hit(conn: sqlite3.Connection, query_hash: str) -> None:
    conn.execute(
        "UPDATE qa_cache SET hit_count = hit_count + 1 WHERE query_hash = ?",
        (query_hash,),
    )


def delete_expired_qa_cache(conn: sqlite3.Connection, now: Optional[datetime] = None) -> int:
    now = now or now_utc()
    cur = conn.execute(
        "DELETE FROM qa_cache WHERE expires_at IS NOT NULL AND expires_at <= ?",
        (_dt(now),),
    )
    return int(cur.rowcount)