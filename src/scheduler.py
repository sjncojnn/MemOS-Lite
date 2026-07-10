"""Rule-based lifecycle scheduler for MemOS-lite.

This is intentionally small: expire TTL, then update hot/warm/cold serving tier.
Cold is a MemoryTier, not a MemoryStatus. Status remains lifecycle-only:
ACTIVE / ARCHIVED / EXPIRED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src import memory_ops
from src.config import Config
from src.memory_store import MemoryStore
from src.qa_cache import QACache
from src.schemas import MemoryStatus, MemoryTier, now_utc


@dataclass
class SchedulerReport:
    checked: int = 0
    marked_expired: int = 0
    marked_hot: int = 0
    marked_warm: int = 0
    marked_cold: int = 0
    tier_changed: int = 0
    qa_cache_cleaned: int = 0
    errors: list[str] = field(default_factory=list)


class MemoryScheduler:
    """Apply Stage-1 memory lifecycle rules outside the request path."""

    def __init__(self, store: MemoryStore, config: Config, cache: Optional[QACache] = None) -> None:
        self.store = store
        self.config = config
        self.cache = cache

    def run(self, now: Optional[datetime] = None, limit: int = 100_000) -> SchedulerReport:
        now = now or now_utc()
        report = SchedulerReport()

        units = self.store.list(status=MemoryStatus.ACTIVE, include_expired=False, limit=limit)
        for unit in units:
            report.checked += 1
            try:
                if unit.ttl_expires_at is not None and unit.ttl_expires_at <= now:
                    self.store.set_status(unit.id, MemoryStatus.EXPIRED, reason="ttl_expired")
                    report.marked_expired += 1
                    continue

                new_tier = memory_ops.choose_tier(unit, self.config)
                if new_tier != unit.tier:
                    self.store.set_tier(unit.id, new_tier, reason="scheduler_tier_update")
                    report.tier_changed += 1
                    if new_tier == MemoryTier.HOT:
                        report.marked_hot += 1
                    elif new_tier == MemoryTier.COLD:
                        report.marked_cold += 1
                    else:
                        report.marked_warm += 1

            except Exception as exc:  # noqa: BLE001 - keep one bad row from stopping the batch
                report.errors.append(f"{unit.id}: {exc}")

        if self.cache is not None:
            try:
                report.qa_cache_cleaned = self.cache.cleanup_expired()
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"qa_cache: {exc}")

        return report
