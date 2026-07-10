"""Lightweight memory operations for MemOS-lite.

This module contains the simple business rules that make the system more than a
plain RAG index: exact duplicate skipping, near-duplicate conflict recording,
TTL expiry, and hot/warm/cold tier updates. It deliberately avoids heavy LLM
judges or complex governance in Stage 1.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Optional

from src.config import Config
from src.memory_store import MemoryStore
from src.schemas import (
    ConflictType,
    MemoryStatus,
    MemoryTier,
    MemoryUnit,
    SourceType,
    compute_content_hash as schema_compute_content_hash,
    normalize_for_hash,
    now_utc,
)


def compute_content_hash(content: str) -> str:
    """Use the canonical schema hash so ingest/db/ops stay consistent."""

    return schema_compute_content_hash(content)


def lexical_similarity(a: str, b: str) -> float:
    """Cheap fallback similarity when embeddings/Chroma are unavailable."""

    left = normalize_for_hash(a)
    right = normalize_for_hash(b)
    if not left or not right:
        return 0.0
    return float(SequenceMatcher(None, left, right).ratio())


def detect_duplicate(store: MemoryStore, candidate: MemoryUnit) -> list[MemoryUnit]:
    """Find exact duplicates by content_hash."""

    candidate.content_hash = candidate.content_hash or compute_content_hash(candidate.content)
    return [unit for unit in store.find_by_content_hash(candidate.content_hash) if unit.id != candidate.id]


def detect_near_duplicates(
    store: MemoryStore,
    candidate: MemoryUnit,
    *,
    threshold: Optional[float] = None,
    top_k: int = 5,
) -> list[tuple[MemoryUnit, float]]:
    """Find near duplicates using vector search when possible, lexical fallback otherwise."""

    threshold = store.config.near_duplicate_threshold if threshold is None else threshold

    if store.embed_fn is not None:
        results = store.search(
            candidate.content,
            top_k=top_k,
            status=MemoryStatus.ACTIVE,
            category=None,
            min_score=threshold,
            touch=False,
        )
        return [
            (result.memory, result.score)
            for result in results
            if result.memory.id != candidate.id and result.memory.content_hash != candidate.content_hash
        ]

    candidates = store.list(status=MemoryStatus.ACTIVE, include_expired=False, limit=1000)
    scored: list[tuple[MemoryUnit, float]] = []
    for unit in candidates:
        if unit.id == candidate.id or unit.content_hash == candidate.content_hash:
            continue
        score = lexical_similarity(candidate.content, unit.content)
        if score >= threshold:
            scored.append((unit, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def detect_conflict(unit_a: MemoryUnit, unit_b: MemoryUnit) -> bool:
    """Very small contradiction heuristic for Stage 1.

    True only when two FAQ-like units have the same normalized question but
    different content. General semantic contradiction should be handled later by
    an LLM-judge or manual admin workflow.
    """

    q_a = normalize_for_hash(str(unit_a.extra_metadata.get("question", "")))
    q_b = normalize_for_hash(str(unit_b.extra_metadata.get("question", "")))
    return bool(q_a and q_a == q_b and unit_a.content_hash != unit_b.content_hash)


def detect_question_conflicts(store: MemoryStore, candidate: MemoryUnit) -> list[MemoryUnit]:
    """Find ACTIVE units whose FAQ question exactly matches candidate's question.

    detect_near_duplicates() only returns pairs whose *overall* text similarity
    clears near_duplicate_threshold. A contradiction (same question, differently
    worded answer) is frequently written in a very different way from the
    original answer, so its whole-string similarity can fall well below that
    threshold and detect_conflict() never gets a chance to look at it. This
    check is independent of text similarity: it matches purely on the
    normalized FAQ question so real contradictions are not silently missed.
    """

    question = normalize_for_hash(str(candidate.extra_metadata.get("question", "")))
    if not question:
        return []

    matches: list[MemoryUnit] = []
    for unit in store.list(status=MemoryStatus.ACTIVE, include_expired=False, limit=1000):
        if unit.id == candidate.id or unit.content_hash == candidate.content_hash:
            continue
        other_question = normalize_for_hash(str(unit.extra_metadata.get("question", "")))
        if other_question and other_question == question:
            matches.append(unit)
    return matches


def resolve_duplicate(store: MemoryStore, keep: MemoryUnit, drop: MemoryUnit) -> None:
    """Archive the duplicate unit and keep the older/canonical one active."""

    store.set_status(drop.id, MemoryStatus.ARCHIVED, reason=f"duplicate_of:{keep.id}")


def add_memory(
    store: MemoryStore,
    unit: MemoryUnit,
    *,
    skip_exact_duplicate: bool = True,
    record_near_duplicate: bool = True,
) -> tuple[Optional[MemoryUnit], list[int]]:
    """Add one unit with exact duplicate skip and near-duplicate conflict records.

    Returns (stored_unit_or_None, conflict_ids). Exact duplicates are skipped by
    default because the Problem Statement explicitly asks for dedup during ingest.
    """

    unit.content_hash = compute_content_hash(unit.content)
    exact_dups = detect_duplicate(store, unit)
    if exact_dups and skip_exact_duplicate:
        return None, []

    near_dups = detect_near_duplicates(store, unit) if record_near_duplicate else []
    question_conflicts = detect_question_conflicts(store, unit) if record_near_duplicate else []
    stored = store.add(unit)

    conflict_ids: list[int] = []
    linked_ids: set[str] = set()
    for other, score in near_dups:
        conflict_type = ConflictType.CONTRADICTION if detect_conflict(stored, other) else ConflictType.NEAR_DUPLICATE
        linked_ids.add(other.id)
        conflict_ids.append(
            store.add_conflict(
                stored.id,
                other.id,
                conflict_type=conflict_type,
                note=f"similarity={score:.3f}",
            )
        )

    for other in question_conflicts:
        if other.id in linked_ids:
            continue
        if detect_conflict(stored, other):
            conflict_ids.append(
                store.add_conflict(
                    stored.id,
                    other.id,
                    conflict_type=ConflictType.CONTRADICTION,
                    note="same_question_different_answer",
                )
            )
    return stored, conflict_ids


def add_many(store: MemoryStore, units: list[MemoryUnit]) -> dict[str, int]:
    """Simple ingest helper used by scripts/tests."""

    stats = {"inserted": 0, "exact_duplicate_skipped": 0, "conflicts": 0}
    for unit in units:
        stored, conflict_ids = add_memory(store, unit)
        if stored is None:
            stats["exact_duplicate_skipped"] += 1
            continue
        stats["inserted"] += 1
        stats["conflicts"] += len(conflict_ids)
    return stats


def is_expired(unit: MemoryUnit) -> bool:
    return unit.ttl_expires_at is not None and unit.ttl_expires_at <= now_utc()


def is_hot(unit: MemoryUnit, config: Config) -> bool:
    """Rule-based hot tier: FAQ or high recent access."""

    if unit.status != MemoryStatus.ACTIVE:
        return False
    if unit.provenance and unit.provenance.source_type == SourceType.FAQ:
        return True
    if "faq" in {tag.lower() for tag in unit.tags}:
        return True
    if unit.access_count < config.hot_access_count_threshold or unit.last_accessed_at is None:
        return False
    return (now_utc() - unit.last_accessed_at).days <= config.hot_window_days


def is_cold(unit: MemoryUnit, config: Config) -> bool:
    """Cold is serving priority, not lifecycle expiration."""

    if unit.status != MemoryStatus.ACTIVE:
        return False
    reference_time = unit.last_accessed_at or unit.created_at
    return (now_utc() - reference_time).days >= config.cold_after_days_no_access


def choose_tier(unit: MemoryUnit, config: Config) -> MemoryTier:
    if is_hot(unit, config):
        return MemoryTier.HOT
    if is_cold(unit, config):
        return MemoryTier.COLD
    return MemoryTier.WARM


def expire_due_memories(store: MemoryStore) -> int:
    """Expire units whose TTL has passed."""

    return store.expire_due()


def update_tiers(store: MemoryStore, *, limit: int = 100_000) -> dict[str, int]:
    """Apply hot/warm/cold tier rules to active memories."""

    stats = {"hot": 0, "warm": 0, "cold": 0, "changed": 0}
    units = store.list(status=MemoryStatus.ACTIVE, include_expired=False, limit=limit)

    for unit in units:
        new_tier = choose_tier(unit, store.config)
        stats[new_tier.value] += 1
        if unit.tier != new_tier:
            store.set_tier(unit.id, new_tier, reason="rule_based_tier_update")
            stats["changed"] += 1
    return stats


def run_lifecycle_maintenance(store: MemoryStore) -> dict[str, int]:
    """One small scheduler-like pass: expire TTL, then update tiers."""

    expired = expire_due_memories(store)
    tier_stats = update_tiers(store)
    return {"expired": expired, **tier_stats}