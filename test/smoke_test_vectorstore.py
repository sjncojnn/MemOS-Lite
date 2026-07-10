"""Smoke test for MemOS-lite: exercises ingest, db, memory_store, memory_ops.

Run multiple times to check idempotency / repeatability of every stage.
"""

from __future__ import annotations

import shutil
import sys
import traceback
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "scripts" else Path.cwd()
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.schemas import (
    MemoryStatus,
    MemoryTier,
    MemoryUnit,
    SourceType,
    now_utc,
)
from src import db
from src.ingest import (
    load_docx_as_units,
    load_faq_xlsx_as_units,
    load_path_as_units,
    clean_text,
)
from src.memory_store import MemoryStore
from src import memory_ops

FAILURES: list[str] = []
PASS_COUNT = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASS_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  [OK] {label}")
    else:
        FAILURES.append(f"{label} :: {detail}")
        print(f"  [FAIL] {label} :: {detail}")


def run_once(run_idx: int) -> None:
    print(f"\n===== SMOKE TEST RUN {run_idx} =====")
    home = Path(f"./.smoke_home_run{run_idx}")
    if home.exists():
        shutil.rmtree(home)
    config = Config(home_dir=home, default_ttl_seconds=600)

    # ---------------------------------------------------------------
    # 1. clean_text
    # ---------------------------------------------------------------
    print("\n-- clean_text --")
    dirty = "  Hello\x00World\r\n\r\n\r\nfoo   bar\t\t \xa0baz  "
    cleaned = clean_text(dirty)
    check("clean_text removes control chars / collapses whitespace",
          "\x00" not in cleaned and "   " not in cleaned, repr(cleaned))
    check("clean_text handles None", clean_text(None) == "")

    # ---------------------------------------------------------------
    # 2. Ingest DOCX
    # ---------------------------------------------------------------
    print("\n-- load_docx_as_units --")
    try:
        docx_units = load_docx_as_units(
            "sample_business.docx",
            category="vay_tien",
            chunk_size_tokens=100,  # force splitting on the long paragraph
            chunk_overlap_tokens=10,
        )
        check("docx produced units", len(docx_units) > 0, f"count={len(docx_units)}")
        check("docx units are MemoryUnit", all(isinstance(u, MemoryUnit) for u in docx_units))
        check("docx section unit exists", any("Vay tiền" in u.content or "vay tien" in u.content.lower() for u in docx_units))
        check("docx table captured", any(u.extra_metadata.get("doc_table") for u in docx_units),
              [u.extra_metadata for u in docx_units])
        long_units = [u for u in docx_units if u.extra_metadata.get("chunk_part", 1) > 1]
        check("long paragraph was split into multiple parts", len(long_units) > 0)
        check("docx units carry provenance", all(u.provenance is not None and u.provenance.source_type == SourceType.DOC for u in docx_units))
        check("docx units default tier WARM", all(u.tier == MemoryTier.WARM for u in docx_units))
    except Exception as exc:
        check("load_docx_as_units did not raise", False, f"{exc}\n{traceback.format_exc()}")
        docx_units = []

    # ---------------------------------------------------------------
    # 3. Ingest FAQ XLSX
    # ---------------------------------------------------------------
    print("\n-- load_faq_xlsx_as_units --")
    try:
        faq_units = load_faq_xlsx_as_units("sample_faq.xlsx", category="faq")
        check("faq produced units (empty row skipped)", len(faq_units) == 5, f"count={len(faq_units)}")
        check("faq units tier HOT", all(u.tier == MemoryTier.HOT for u in faq_units))
        check("faq units carry question/answer metadata",
              all("question" in u.extra_metadata and "answer" in u.extra_metadata for u in faq_units))
        exact_dup_hashes = [u.content_hash for u in faq_units]
        check("faq contains an exact duplicate pair (by design)",
              len(exact_dup_hashes) != len(set(exact_dup_hashes)))
    except Exception as exc:
        check("load_faq_xlsx_as_units did not raise", False, f"{exc}\n{traceback.format_exc()}")
        faq_units = []

    # ---------------------------------------------------------------
    # 4. load_path_as_units (directory dispatch)
    # ---------------------------------------------------------------
    print("\n-- load_path_as_units (directory) --")
    try:
        src_dir = Path("ingest_dir_test")
        src_dir.mkdir(exist_ok=True)
        shutil.copy("sample_business.docx", src_dir / "sample_business.docx")
        shutil.copy("sample_faq.xlsx", src_dir / "sample_faq.xlsx")
        all_units = load_path_as_units(src_dir, config=config, category="uncategorized")
        check("directory ingest picked up both files", len(all_units) > 0, f"count={len(all_units)}")
    except Exception as exc:
        check("load_path_as_units did not raise", False, f"{exc}\n{traceback.format_exc()}")

    # ---------------------------------------------------------------
    # 5. DB layer directly (schema, CRUD, lifecycle log, conflicts, qa_cache)
    # ---------------------------------------------------------------
    print("\n-- db.py direct --")
    try:
        db.init_db(config.sqlite_path)
        db.init_db(config.sqlite_path)  # idempotent re-init
        check("init_db idempotent", True)

        with db.connect(config.sqlite_path) as conn:
            u = MemoryUnit(content="Test đơn vị tri thức", source="unit_test", category="test")
            db.insert_memory_unit(conn, u)
            fetched = db.get_memory_unit(conn, u.id)
            check("insert/get memory unit roundtrip", fetched is not None and fetched.content == u.content)

            db.touch_access(conn, u.id)
            touched = db.get_memory_unit(conn, u.id)
            check("touch_access increments access_count", touched.access_count == 1, touched.access_count)
            check("touch_access sets last_accessed_at", touched.last_accessed_at is not None)

            updated = db.update_memory_content(conn, u.id, "Nội dung đã cập nhật", summary="tóm tắt")
            check("update_memory_content increments version", updated.version == 2, updated.version)
            check("update_memory_content changes content_hash", updated.content_hash != u.content_hash)

            db.update_status(conn, u.id, MemoryStatus.ARCHIVED, reason="test_archive")
            archived = db.get_memory_unit(conn, u.id)
            check("update_status changes status", archived.status == MemoryStatus.ARCHIVED)

            db.update_tier(conn, u.id, MemoryTier.HOT, reason="test_tier")
            hot = db.get_memory_unit(conn, u.id)
            check("update_tier changes tier", hot.tier == MemoryTier.HOT)

            log = db.get_lifecycle_log(conn, u.id)
            check("lifecycle_log has multiple events", len(log) >= 3, len(log))

            u2 = MemoryUnit(content="Đơn vị thứ hai", source="unit_test", category="test")
            db.insert_memory_unit(conn, u2)
            conflict_id = db.insert_conflict(conn, u.id, u2.id, note="similarity=0.95")
            conflict_id_2 = db.insert_conflict(conn, u2.id, u.id, note="dup insert reversed order")
            check("insert_conflict is symmetric (A,B)==(B,A)", conflict_id == conflict_id_2,
                  (conflict_id, conflict_id_2))

            conflicts = db.list_unresolved_conflicts(conn)
            check("list_unresolved_conflicts returns the record", any(c.id == conflict_id for c in conflicts))

            db.resolve_conflict(conn, conflict_id, "kept_a", note="resolved in test")
            resolved = db.list_conflicts(conn, resolution="kept_a")
            check("resolve_conflict updates resolution", any(c.id == conflict_id for c in resolved))

            try:
                db.insert_conflict(conn, u.id, u.id)
                check("insert_conflict rejects self-pair", False, "did not raise")
            except ValueError:
                check("insert_conflict rejects self-pair", True)

            qhash = "q-hash-test"
            db.upsert_qa_cache(conn, qhash, "câu hỏi test", "câu trả lời test",
                                expires_at=now_utc() + timedelta(seconds=100),
                                retrieved_memory_ids=[u.id])
            cache_entry = db.get_valid_qa_cache(conn, qhash)
            check("qa_cache stores and retrieves valid entry", cache_entry is not None and cache_entry.hit_count == 1)

            db.upsert_qa_cache(conn, "expired-hash", "q2", "a2", expires_at=now_utc() - timedelta(seconds=1))
            expired_entry = db.get_valid_qa_cache(conn, "expired-hash")
            check("qa_cache returns None for expired entry", expired_entry is None)

            removed = db.delete_expired_qa_cache(conn)
            check("delete_expired_qa_cache removes expired rows", removed >= 1, removed)

            db.delete_memory_unit(conn, u2.id)
            check("delete_memory_unit hard deletes", db.get_memory_unit(conn, u2.id) is None)
    except Exception as exc:
        check("db.py direct block did not raise", False, f"{exc}\n{traceback.format_exc()}")

    # ---------------------------------------------------------------
    # 6. MemoryStore (no embed_fn -> SQLite-only path)
    # ---------------------------------------------------------------
    print("\n-- MemoryStore (no embed_fn) --")
    store = None
    try:
        store = MemoryStore(config, embed_fn=None)
        check("MemoryStore constructs without embed_fn", store is not None)

        unit = MemoryUnit(content="Nội dung không có embedding", source="store_test", category="test")
        stored = store.add(unit)
        check("store.add persists unit", stored.id == unit.id)
        check("store.add applies default TTL", stored.ttl_expires_at is not None)

        fetched = store.get(stored.id)
        check("store.get returns unit", fetched is not None and fetched.content == stored.content)

        stored.content = "Nội dung sau khi sửa trực tiếp"
        store.update(stored)
        refetched = store.get(stored.id)
        check("store.update persists content change", refetched.content == "Nội dung sau khi sửa trực tiếp")

        versioned = store.update_content(stored.id, "Nội dung phiên bản mới", summary="tóm tắt mới")
        check("store.update_content increments version", versioned.version >= 2, versioned.version)

        by_hash = store.find_by_content_hash(versioned.content_hash)
        check("store.find_by_content_hash finds the unit", any(u.id == versioned.id for u in by_hash))

        store.touch_access(stored.id)
        touched = store.get(stored.id)
        check("store.touch_access increments access_count", touched.access_count == 1, touched.access_count)

        store.set_tier(stored.id, MemoryTier.COLD, reason="test")
        check("store.set_tier updates tier", store.get(stored.id).tier == MemoryTier.COLD)

        store.set_status(stored.id, MemoryStatus.ARCHIVED, reason="test")
        check("store.set_status updates status", store.get(stored.id).status == MemoryStatus.ARCHIVED)

        listing = store.list(status=None, include_expired=True, limit=1000)
        check("store.list returns units", len(listing) > 0, len(listing))

        empty_search = store.search("bất kỳ câu hỏi nào", top_k=3)
        check("store.search returns empty list without embed_fn", empty_search == [])

        store.delete(stored.id, hard=False)
        check("store.delete (soft) archives", store.get(stored.id).status == MemoryStatus.ARCHIVED)

        hard_unit = store.add(MemoryUnit(content="Sẽ bị xóa cứng", source="store_test", category="test"))
        store.delete(hard_unit.id, hard=True)
        check("store.delete (hard) removes row", store.get(hard_unit.id) is None)

    except Exception as exc:
        check("MemoryStore block did not raise", False, f"{exc}\n{traceback.format_exc()}")

    # ---------------------------------------------------------------
    # 7. memory_ops: dedup / conflict / add_many / tiering / lifecycle
    # ---------------------------------------------------------------
    print("\n-- memory_ops --")
    try:
        if store is None:
            store = MemoryStore(config, embed_fn=None)

        fresh_faq_units = load_faq_xlsx_as_units("sample_faq.xlsx", category="faq")
        stats = memory_ops.add_many(store, fresh_faq_units)
        check("add_many inserted some units", stats["inserted"] > 0, stats)
        check("add_many skipped the exact duplicate row", stats["exact_duplicate_skipped"] >= 1, stats)
        check("add_many recorded at least one conflict (contradiction/near-dup)",
              stats["conflicts"] >= 1, stats)

        # Run add_many again with a FRESH load (new ids, same content) -- everything
        # should now be recognized as an exact duplicate by content_hash.
        reloaded_faq_units = load_faq_xlsx_as_units("sample_faq.xlsx", category="faq")
        stats2 = memory_ops.add_many(store, reloaded_faq_units)
        check("re-running add_many skips all as exact duplicates",
              stats2["inserted"] == 0 and stats2["exact_duplicate_skipped"] == len(reloaded_faq_units),
              stats2)

        all_active = store.list(status=MemoryStatus.ACTIVE, include_expired=False, limit=1000)
        faq_active = [u for u in all_active if u.provenance and u.provenance.source_type == SourceType.FAQ]
        check("faq units are ACTIVE after add_many", len(faq_active) >= 3, len(faq_active))

        conflicts = store.list_conflicts()
        check("store.list_conflicts returns records after add_many", len(conflicts) >= 1, len(conflicts))

        contradiction_hits = [c for c in conflicts if c.conflict_type.value == "contradiction"]
        check("detect_conflict found the same-question-different-answer FAQ row",
              len(contradiction_hits) >= 1, [c.note for c in conflicts])

        if len(faq_active) >= 2:
            keep, drop = faq_active[0], faq_active[1]
            memory_ops.resolve_duplicate(store, keep, drop)
            check("resolve_duplicate archives the dropped unit",
                  store.get(drop.id).status == MemoryStatus.ARCHIVED)

        # Tiering
        old_unit = MemoryUnit(
            content="Đơn vị cũ để test cold tier",
            source="tier_test",
            category="test",
            tier=MemoryTier.WARM,
        )
        stored_old = store.add(old_unit)
        with db.connect(config.sqlite_path) as conn:
            far_past = (now_utc() - timedelta(days=999)).isoformat()
            conn.execute(
                "UPDATE memory_units SET created_at = ?, updated_at = ? WHERE id = ?",
                (far_past, far_past, stored_old.id),
            )
            conn.commit()

        check("is_cold true for very old unit", memory_ops.is_cold(store.get(stored_old.id), config))
        tier_stats = memory_ops.update_tiers(store)
        check("update_tiers moved the old unit to cold", store.get(stored_old.id).tier == MemoryTier.COLD, tier_stats)
        check("update_tiers reports changed>=1", tier_stats["changed"] >= 1, tier_stats)

        # TTL expiry
        ttl_unit = MemoryUnit(
            content="Đơn vị hết hạn để test TTL",
            source="ttl_test",
            category="test",
            ttl_expires_at=now_utc() - timedelta(seconds=5),
        )
        stored_ttl = store.add(ttl_unit, apply_default_ttl=False)
        expired_count = memory_ops.expire_due_memories(store)
        check("expire_due_memories expires the due unit", expired_count >= 1, expired_count)
        check("expired unit status == EXPIRED", store.get(stored_ttl.id).status == MemoryStatus.EXPIRED)
        check("expired unit is_available == False", store.get(stored_ttl.id).is_available is False)

        maint = memory_ops.run_lifecycle_maintenance(store)
        check("run_lifecycle_maintenance returns combined stats",
              set(maint.keys()) == {"expired", "hot", "warm", "cold", "changed"}, maint)

        # is_hot rule: FAQ should always be hot
        faq_sample = next((u for u in store.list(status=MemoryStatus.ACTIVE, limit=1000)
                            if u.provenance and u.provenance.source_type == SourceType.FAQ), None)
        if faq_sample:
            check("is_hot true for FAQ-sourced unit", memory_ops.is_hot(faq_sample, config))

    except Exception as exc:
        check("memory_ops block did not raise", False, f"{exc}\n{traceback.format_exc()}")

    # ---------------------------------------------------------------
    # 8. lexical_similarity sanity
    # ---------------------------------------------------------------
    print("\n-- lexical_similarity --")
    try:
        sim_same = memory_ops.lexical_similarity("xin chào bạn", "xin chào bạn")
        sim_diff = memory_ops.lexical_similarity("xin chào bạn", "thời tiết hôm nay thế nào")
        check("lexical_similarity identical == 1.0", sim_same == 1.0, sim_same)
        check("lexical_similarity unrelated text is low", sim_diff < 0.5, sim_diff)
        check("lexical_similarity empty string -> 0.0", memory_ops.lexical_similarity("", "abc") == 0.0)
    except Exception as exc:
        check("lexical_similarity block did not raise", False, f"{exc}\n{traceback.format_exc()}")

    shutil.rmtree(home, ignore_errors=True)
    shutil.rmtree(Path("ingest_dir_test"), ignore_errors=True)


if __name__ == "__main__":
    for i in range(1, 4):  # run 3 times to check repeatability / idempotency
        run_once(i)

    print(f"\n\n===== SUMMARY: {PASS_COUNT} passed, {len(FAILURES)} failed =====")
    if FAILURES:
        print("Failures:")
        for f in FAILURES:
            print(f" - {f}")
        sys.exit(1)
    else:
        print("All smoke tests passed.")