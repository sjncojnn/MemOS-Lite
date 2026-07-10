"""Smoke tests for MemOS-lite retriever.py and qa_cache.py.

Run from the project root after copying the updated files into src/:

    python tests/test_retriever_qa_cache_smoke.py

or with pytest:

    pytest -q tests/test_retriever_qa_cache_smoke.py

The tests do not call Ollama and do not require Chroma. They use in-memory fake
stores/vector indexes to verify the retrieval/cache policies.
"""

from __future__ import annotations

import math
import re
import sys
import unicodedata
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional, Sequence

# Allow running as `python tests/test_...py` from repo root or tests/.
PROJECT_ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "tests" else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import db  # noqa: E402
from src.config import Config  # noqa: E402
from src.qa_cache import QACache  # noqa: E402
import src.qa_cache as qa_cache_module  # noqa: E402
from src.retriever import MemoryRetriever, _tokens  # noqa: E402
from src.schemas import (  # noqa: E402
    CacheEntry,
    ConflictRecord,
    ConflictResolution,
    ConflictType,
    MemoryStatus,
    MemoryTier,
    MemoryUnit,
    Provenance,
    RetrievedMemory,
    SourceType,
    compute_query_hash,
    now_utc,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _make_config(tmp: str | Path | None = None) -> Config:
    cfg = Config(home_dir=Path(tmp) if tmp else Path(TemporaryDirectory().name))
    # Tight values make smoke tests deterministic and fast.
    cfg.top_k = 3
    cfg.max_context_chars = 1200
    cfg.min_retrieval_score = 0.0
    cfg.qa_cache_ttl_seconds = 100
    cfg.qa_semantic_cache_enabled = True
    cfg.qa_semantic_answer_threshold = 0.94
    cfg.qa_semantic_sensitive_threshold = 0.96
    cfg.qa_semantic_retrieval_threshold = 0.90
    cfg.qa_semantic_min_memory_overlap = 0.60
    cfg.retrieval_candidate_multiplier = 4
    cfg.retrieval_lexical_pool_limit = 1000
    cfg.retrieval_neighbor_expansion = True
    cfg.retrieval_exclude_conflicts = True
    cfg.retrieval_semantic_candidate_min_score = 0.0
    return cfg


def _prov(source_type: SourceType, ref: str = "") -> Provenance:
    return Provenance(source_type=source_type, source_path=f"fake.{source_type.value}", source_ref=ref)


def _unit(
    memory_id: str,
    content: str,
    *,
    category: str = "uncategorized",
    tags: Optional[list[str]] = None,
    source: str = "source.txt",
    source_type: Optional[SourceType] = SourceType.DOC,
    tier: MemoryTier = MemoryTier.WARM,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    question: str = "",
    answer: str = "",
    heading: str = "",
) -> MemoryUnit:
    extra: dict[str, Any] = {}
    if question:
        extra["question"] = question
    if answer:
        extra["answer"] = answer
    if heading:
        extra["doc_heading"] = heading
    return MemoryUnit(
        id=memory_id,
        content=content,
        source=source,
        category=category,
        tags=tags or [],
        status=status,
        tier=tier,
        provenance=_prov(source_type, ref=heading or memory_id) if source_type else None,
        extra_metadata=extra,
    )


class FakeMemoryStore:
    """Small in-memory substitute for MemoryStore used by retriever/cache tests."""

    def __init__(
        self,
        config: Config,
        units: Sequence[MemoryUnit],
        *,
        semantic_results: Optional[list[RetrievedMemory]] = None,
        conflicts: Optional[list[ConflictRecord]] = None,
    ) -> None:
        self.config = config
        self.embed_fn = None
        self.units = {u.id: u for u in units}
        self.semantic_results = semantic_results or []
        self.conflicts = conflicts or []
        self.touched: list[str] = []
        self.expire_due_calls = 0
        self.search_calls: list[dict[str, Any]] = []

    def expire_due(self) -> int:
        self.expire_due_calls += 1
        expired = 0
        for unit in self.units.values():
            if unit.ttl_expires_at and unit.ttl_expires_at <= now_utc() and unit.status != MemoryStatus.EXPIRED:
                unit.status = MemoryStatus.EXPIRED
                expired += 1
        return expired

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
        self.search_calls.append(
            {
                "query": query,
                "top_k": top_k,
                "status": status,
                "category": category,
                "tier": tier,
                "min_score": min_score,
                "touch": touch,
            }
        )
        status_value = status.value if hasattr(status, "value") else status
        tier_value = tier.value if hasattr(tier, "value") else tier
        results: list[RetrievedMemory] = []
        for result in self.semantic_results:
            unit = result.memory
            if status_value is not None and unit.status.value != status_value:
                continue
            if category is not None and unit.category != category:
                continue
            if tier_value is not None and unit.tier.value != tier_value:
                continue
            if min_score is not None and result.score < min_score:
                continue
            results.append(result)
        return results[: top_k or len(results)]

    def list(
        self,
        status: Optional[MemoryStatus | str] = None,
        category: Optional[str] = None,
        tier: Optional[MemoryTier | str] = None,
        include_expired: bool = True,
        limit: int = 100,
    ) -> list[MemoryUnit]:
        status_value = status.value if hasattr(status, "value") else status
        tier_value = tier.value if hasattr(tier, "value") else tier
        rows: list[MemoryUnit] = []
        for unit in self.units.values():
            if status_value is not None and unit.status.value != status_value:
                continue
            if not include_expired and not unit.is_available:
                continue
            if category is not None and unit.category != category:
                continue
            if tier_value is not None and unit.tier.value != tier_value:
                continue
            rows.append(unit)
        return rows[:limit]

    def get(self, memory_id: str) -> Optional[MemoryUnit]:
        # Deliberately raw-id only. This lets retrieve_by_ids() prove it parses
        # version-aware refs like "mem1::v2::hash" before calling store.get().
        return self.units.get(memory_id)

    def touch_access(self, memory_id: str) -> None:
        unit = self.units[memory_id]
        unit.access_count += 1
        unit.last_accessed_at = now_utc()
        self.touched.append(memory_id)

    def list_conflicts(
        self,
        resolution: Optional[ConflictResolution | str] = None,
        limit: int = 100,
    ) -> list[ConflictRecord]:
        resolution_value = resolution.value if hasattr(resolution, "value") else resolution
        rows = [c for c in self.conflicts if resolution_value is None or c.resolution.value == resolution_value]
        return rows[:limit]


class FakeVectorStore:
    """In-memory stand-in for src.vector_store.VectorStore used by QACache."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        for idx, row_id in enumerate(ids):
            self.rows[row_id] = {
                "embedding": embeddings[idx],
                "document": documents[idx],
                "metadata": (metadatas or [{} for _ in ids])[idx],
            }

    def query(self, query_embedding: list[float], top_k: int = 5, where: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        for row_id, row in self.rows.items():
            sim = _cosine(query_embedding, row["embedding"])
            hits.append(
                {
                    "id": row_id,
                    "document": row["document"],
                    "metadata": row["metadata"],
                    "distance": 1.0 - sim,
                }
            )
        hits.sort(key=lambda h: float(h["distance"]))
        return hits[:top_k]

    def delete(self, ids: list[str]) -> None:
        for row_id in ids:
            self.rows.pop(row_id, None)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def _norm_test_text(text: str) -> str:
    text = str(text or "").lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", text)


def fake_embed(text: str) -> list[float]:
    """Tiny deterministic embedding with a few Vietnamese synonym buckets."""

    n = _norm_test_text(text)
    features = [
        int(any(x in n for x in ["thanh toan", "dong", "nop", "tra tien"])),
        int(any(x in n for x in ["hoa don", "tien dien", "tien nuoc"])),
        int("dien" in n),
        int("nuoc" in n),
        int("phi" in n or "gia" in n or "bao nhieu" in n),
        int("han muc" in n),
    ]
    return [float(x) for x in features]


# ---------------------------------------------------------------------------
# Retriever tests
# ---------------------------------------------------------------------------


def test_retriever_filters_vietnamese_stopwords_and_uses_query_coverage_for_long_doc_chunks() -> None:
    cfg = _make_config()
    long_relevant_doc = _unit(
        "doc_long",
        " ".join(["nội dung chung"] * 160)
        + " Khách hàng thanh toán hóa đơn điện bằng cách nhập mã khách hàng và xác nhận giao dịch.",
        category="hoa_don_dien",
        source="hoa_don_dien.docx",
        source_type=SourceType.DOC,
        heading="Thanh toán hóa đơn điện",
    )
    stopword_noise = _unit(
        "noise",
        "khách hàng có thể như thế nào không " * 80,
        category="hoi_dap_chung",
        source="noise.docx",
        source_type=SourceType.DOC,
    )
    store = FakeMemoryStore(cfg, [long_relevant_doc, stopword_noise])
    retriever = MemoryRetriever(store)

    results = retriever.retrieve("Khách hàng có thể thanh toán hóa đơn điện như thế nào?", top_k=2)

    assert results, "Expected lexical fallback to retrieve a long doc chunk."
    assert results[0].memory.id == "doc_long"
    assert "noise" not in [r.memory.id for r in results], "Vietnamese stopword noise should not dominate lexical score."
    assert store.expire_due_calls == 1
    assert store.touched == ["doc_long"]


def test_retriever_category_is_soft_boost_by_default_and_strict_only_when_requested() -> None:
    cfg = _make_config()
    cross_category_answer = _unit(
        "cross_cat",
        "Hóa đơn điện được thanh toán trong mục hóa đơn bằng mã khách hàng.",
        category="tong_quan",
        source="tong_quan.docx",
        source_type=SourceType.DOC,
        heading="Hóa đơn điện",
    )
    store = FakeMemoryStore(cfg, [cross_category_answer])
    retriever = MemoryRetriever(store)

    soft = retriever.retrieve("thanh toán hóa đơn điện", category="hoa_don_dien", category_is_strict=False, touch=False)
    strict = retriever.retrieve("thanh toán hóa đơn điện", category="hoa_don_dien", category_is_strict=True, touch=False)

    assert [r.memory.id for r in soft] == ["cross_cat"], "Soft category should not hide relevant memories in another category."
    assert strict == [], "Strict category should restrict both vector and lexical candidate pools."


def test_retriever_excludes_unresolved_conflict_by_default() -> None:
    cfg = _make_config()
    conflicted = _unit(
        "limit_10m",
        "Hạn mức chuyển tiền là 10 triệu đồng.",
        category="chuyen_tien",
        source_type=SourceType.FAQ,
        tier=MemoryTier.HOT,
        question="Hạn mức chuyển tiền là bao nhiêu?",
        answer="10 triệu đồng",
    )
    safe = _unit(
        "safe_doc",
        "Thông tin chuyển tiền liên ngân hàng và hạn mức cần xem theo biểu phí hiện hành.",
        category="chuyen_tien",
        source_type=SourceType.DOC,
    )
    conflict = ConflictRecord(
        id=1,
        memory_id_a="limit_10m",
        memory_id_b="limit_20m",
        conflict_type=ConflictType.CONTRADICTION,
        resolution=ConflictResolution.UNRESOLVED,
    )
    store = FakeMemoryStore(cfg, [conflicted, safe], conflicts=[conflict])
    retriever = MemoryRetriever(store)

    results = retriever.retrieve("hạn mức chuyển tiền là bao nhiêu", top_k=3, touch=False)
    ids = [r.memory.id for r in results]

    assert "limit_10m" not in ids, "Unresolved conflicting memory should not enter QA context by default."
    assert "safe_doc" in ids


def test_retriever_separates_semantic_candidate_threshold_from_final_min_score() -> None:
    cfg = _make_config()
    cfg.retrieval_semantic_candidate_min_score = 0.0
    semantic_low = _unit("semantic_low", "thanh toán hóa đơn điện", source_type=SourceType.DOC)
    store = FakeMemoryStore(cfg, [semantic_low], semantic_results=[RetrievedMemory(semantic_low, score=0.20)])
    retriever = MemoryRetriever(store)

    retriever.retrieve("thanh toán hóa đơn điện", min_score=0.50, touch=False)

    assert store.search_calls, "Retriever should call store.search for semantic candidates."
    assert store.search_calls[-1]["min_score"] == 0.0, "Semantic candidate threshold must not reuse final min_score."


def test_retriever_retrieve_by_ids_accepts_version_aware_refs() -> None:
    cfg = _make_config()
    unit = _unit("mem1", "Nội dung cần lấy lại từ retrieval cache.")
    store = FakeMemoryStore(cfg, [unit])
    retriever = MemoryRetriever(store)

    results = retriever.retrieve_by_ids(["mem1::v2::habc", "mem1::v2::habc"])

    assert len(results) == 1
    assert results[0].memory.id == "mem1"
    assert store.touched == ["mem1"]


def test_retriever_build_context_compacts_faq_and_respects_max_chars() -> None:
    cfg = _make_config()
    faq = _unit(
        "faq1",
        "raw content should be compacted",
        source="faq.xlsx",
        source_type=SourceType.FAQ,
        tier=MemoryTier.HOT,
        question="Khách hàng thanh toán hóa đơn điện thế nào?",
        answer="Vào mục Hóa đơn điện, nhập mã khách hàng và xác nhận.",
    )
    store = FakeMemoryStore(cfg, [faq])
    retriever = MemoryRetriever(store)

    context = retriever.build_context([RetrievedMemory(faq, score=0.9, rank=1)], max_chars=260)

    assert "Q: Khách hàng thanh toán hóa đơn điện thế nào?" in context
    assert "A: Vào mục Hóa đơn điện" in context
    assert "source_type=faq" in context
    assert len(context) <= 260


# ---------------------------------------------------------------------------
# QA cache tests
# ---------------------------------------------------------------------------


def _make_cache_with_fake_vector(tmp: str | Path, embed_fn=fake_embed) -> QACache:
    # Patch the VectorStore symbol imported inside src.qa_cache, not src.vector_store.
    qa_cache_module.VectorStore = FakeVectorStore  # type: ignore[assignment]
    cfg = _make_config(tmp)
    return QACache(cfg, embed_fn=embed_fn)


def test_qa_cache_exact_hit_and_invalid_memory_ref_guard() -> None:
    with TemporaryDirectory() as tmp:
        cfg = _make_config(tmp)
        active = _unit("mem_active", "Active answer source")
        store = FakeMemoryStore(cfg, [active])
        cache = QACache(cfg)

        cache.set("Thanh toán hóa đơn điện thế nào?", "Câu trả lời chuẩn", ["mem_active"])
        hit = cache.get_exact("  Thanh toán hóa đơn điện thế nào?  ", store=store)
        assert hit is not None
        assert hit.answer == "Câu trả lời chuẩn"
        assert hit.hit_count == 1

        active.status = MemoryStatus.ARCHIVED
        assert cache.get_exact("Thanh toán hóa đơn điện thế nào?", store=store) is None


def test_qa_cache_semantic_answer_requires_high_similarity_active_refs_and_memory_overlap() -> None:
    with TemporaryDirectory() as tmp:
        cache = _make_cache_with_fake_vector(tmp)
        faq = _unit("faq_electric", "FAQ electric", source_type=SourceType.FAQ, tier=MemoryTier.HOT)
        doc = _unit("doc_electric", "DOC electric", source_type=SourceType.DOC)
        other = _unit("other", "Other memory", source_type=SourceType.DOC)
        store = FakeMemoryStore(cache.config, [faq, doc, other])

        cache.set(
            "Khách hàng thanh toán hóa đơn điện như thế nào?",
            "Dùng mục hóa đơn điện, nhập mã khách hàng và xác nhận.",
            ["faq_electric", "doc_electric"],
        )

        semantic = cache.get_semantic_answer(
            "Khách hàng đóng tiền điện ra sao?",
            store=store,
            current_memory_ids=["faq_electric", "doc_electric"],
        )
        no_overlap = cache.get_semantic_answer(
            "Khách hàng đóng tiền điện ra sao?",
            store=store,
            current_memory_ids=["other"],
        )

        assert semantic is not None
        assert semantic.answer.startswith("Dùng mục hóa đơn điện")
        assert no_overlap is None, "Semantic answer cache must be rejected when retrieval signature does not overlap."


def test_qa_cache_semantic_answer_blocks_unresolved_conflict() -> None:
    with TemporaryDirectory() as tmp:
        cache = _make_cache_with_fake_vector(tmp)
        faq = _unit("faq_conflict", "FAQ conflict", source_type=SourceType.FAQ, tier=MemoryTier.HOT)
        conflict = ConflictRecord(
            id=1,
            memory_id_a="faq_conflict",
            memory_id_b="faq_other",
            conflict_type=ConflictType.CONTRADICTION,
            resolution=ConflictResolution.UNRESOLVED,
        )
        store = FakeMemoryStore(cache.config, [faq], conflicts=[conflict])

        cache.set("Hạn mức chuyển tiền là bao nhiêu?", "10 triệu", ["faq_conflict"])
        result = cache.get_semantic_answer(
            "Cho tôi biết hạn mức chuyển tiền?",
            store=store,
            current_memory_ids=["faq_conflict"],
        )

        assert result is None, "QA cache must not reuse answers based on unresolved conflicting memories."


def test_qa_cache_retrieval_hint_is_weaker_than_semantic_answer_cache() -> None:
    with TemporaryDirectory() as tmp:
        cfg = _make_config(tmp)
        cache = QACache(cfg)
        faq = _unit("faq_hint", "FAQ hint", source_type=SourceType.FAQ, tier=MemoryTier.HOT)
        store = FakeMemoryStore(cfg, [faq])
        entry = CacheEntry(
            query_hash=compute_query_hash("cached query"),
            query="cached query",
            answer="cached answer",
            retrieved_memory_ids=["faq_hint"],
        )
        # Avoid dependence on fake vector similarity here; test policy thresholds directly.
        cache.embed_fn = fake_embed
        cache.vector_store = FakeVectorStore()
        cache._semantic_enabled = lambda: True  # type: ignore[method-assign]
        cache._semantic_candidates = lambda query, top_k: [(entry, 0.91)]  # type: ignore[method-assign]

        hint = cache.get_retrieval_hint("similar query", store=store)
        answer = cache.get_semantic_answer("similar query", store=store, current_memory_ids=["faq_hint"])

        assert hint == ["faq_hint"], "Similarity 0.91 should be enough for retrieval hint."
        assert answer is None, "Similarity 0.91 should not be enough to reuse final answer."


def test_qa_cache_sensitive_queries_use_higher_answer_threshold() -> None:
    with TemporaryDirectory() as tmp:
        cfg = _make_config(tmp)
        cache = QACache(cfg)
        faq = _unit("fee_faq", "Fee FAQ", source_type=SourceType.FAQ, tier=MemoryTier.HOT)
        store = FakeMemoryStore(cfg, [faq])
        entry = CacheEntry(
            query_hash=compute_query_hash("phí cached"),
            query="phí cached",
            answer="phí là 1.000đ",
            retrieved_memory_ids=["fee_faq"],
        )
        cache.embed_fn = fake_embed
        cache.vector_store = FakeVectorStore()
        cache._semantic_enabled = lambda: True  # type: ignore[method-assign]
        cache._semantic_candidates = lambda query, top_k: [(entry, 0.95)]  # type: ignore[method-assign]

        normal = cache.get_semantic_answer("thanh toán hóa đơn điện", store=store, current_memory_ids=["fee_faq"])
        sensitive = cache.get_semantic_answer("phí thanh toán hóa đơn điện bao nhiêu", store=store, current_memory_ids=["fee_faq"])

        assert normal is not None, "0.95 is above normal semantic answer threshold."
        assert sensitive is None, "Sensitive fee/amount queries require the stricter threshold."


def test_qa_cache_cleanup_expired_rows() -> None:
    with TemporaryDirectory() as tmp:
        cfg = _make_config(tmp)
        db.init_db(cfg.sqlite_path)
        expired_hash = compute_query_hash("expired query")
        valid_hash = compute_query_hash("valid query")
        with db.connect(cfg.sqlite_path) as conn:
            db.upsert_qa_cache(
                conn,
                expired_hash,
                "expired query",
                "old answer",
                expires_at=now_utc() - timedelta(seconds=1),
                retrieved_memory_ids=[],
            )
            db.upsert_qa_cache(
                conn,
                valid_hash,
                "valid query",
                "valid answer",
                expires_at=now_utc() + timedelta(seconds=60),
                retrieved_memory_ids=[],
            )
        cache = QACache(cfg)

        deleted = cache.cleanup_expired()
        with db.connect(cfg.sqlite_path) as conn:
            expired = db.get_qa_cache(conn, expired_hash)
            valid = db.get_qa_cache(conn, valid_hash)

        assert deleted == 1
        assert expired is None
        assert valid is not None



def test_retriever_matches_unaccented_query_to_accented_vietnamese_content() -> None:
    cfg = _make_config()
    unit = _unit(
        "accented_doc",
        "Khách hàng thanh toán hóa đơn điện bằng mã khách hàng trên ứng dụng.",
        category="hoa_don_dien",
        source_type=SourceType.DOC,
    )
    store = FakeMemoryStore(cfg, [unit])
    retriever = MemoryRetriever(store)

    results = retriever.retrieve("thanh toan hoa don dien bang ma khach hang", top_k=1, touch=False)

    assert results
    assert results[0].memory.id == "accented_doc"


def test_qa_cache_exact_invalid_memory_ref_does_not_increment_hit_count() -> None:
    with TemporaryDirectory() as tmp:
        cfg = _make_config(tmp)
        unit = _unit("stale_mem", "Source that later becomes archived")
        store = FakeMemoryStore(cfg, [unit])
        cache = QACache(cfg)

        cache.set("câu hỏi", "answer", ["stale_mem"])
        unit.status = MemoryStatus.ARCHIVED

        assert cache.get_exact("câu hỏi", store=store) is None
        with db.connect(cfg.sqlite_path) as conn:
            row = db.get_qa_cache(conn, compute_query_hash("câu hỏi"))
        assert row is not None
        assert row.hit_count == 0, "Rejected cache entries should not be counted as cache hits."


def test_qa_cache_version_aware_refs_validate_raw_id_version_and_hash_prefix() -> None:
    with TemporaryDirectory() as tmp:
        cfg = _make_config(tmp)
        unit = _unit("versioned_mem", "Versioned source")
        unit.version = 2
        ref = f"{unit.id}::v2::h{unit.content_hash[:8]}"
        store = FakeMemoryStore(cfg, [unit])
        cache = QACache(cfg)

        cache.set("versioned query", "versioned answer", [ref])
        assert cache.get_exact("versioned query", store=store) is not None

        unit.version = 3
        assert cache.get_exact("versioned query", store=store) is None


def test_qa_cache_semantic_overlap_accepts_version_refs_against_raw_current_ids() -> None:
    with TemporaryDirectory() as tmp:
        cache = _make_cache_with_fake_vector(tmp)
        unit = _unit("faq_versioned", "FAQ source", source_type=SourceType.FAQ, tier=MemoryTier.HOT)
        unit.version = 1
        ref = f"{unit.id}::v1::h{unit.content_hash[:8]}"
        store = FakeMemoryStore(cache.config, [unit])

        cache.set("Khách hàng thanh toán hóa đơn điện như thế nào?", "answer", [ref])
        hit = cache.get_semantic_answer(
            "Khách hàng đóng tiền điện ra sao?",
            store=store,
            current_memory_ids=["faq_versioned"],
        )

        assert hit is not None
        assert hit.answer == "answer"


def test_qa_cache_invalidate_deletes_sqlite_row_and_vector_row() -> None:
    with TemporaryDirectory() as tmp:
        cache = _make_cache_with_fake_vector(tmp)
        store = FakeMemoryStore(cache.config, [_unit("mem", "source")])

        cache.set("query to remove", "answer", ["mem"])
        query_hash = compute_query_hash("query to remove")
        assert cache.vector_store is not None
        assert query_hash in cache.vector_store.rows

        cache.invalidate("query to remove")

        assert cache.get_exact("query to remove", store=store) is None
        assert query_hash not in cache.vector_store.rows

# ---------------------------------------------------------------------------
# Minimal runner for `python tests/test_...py`
# ---------------------------------------------------------------------------


def _run_all() -> None:
    tests = [(name, obj) for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001 - smoke runner should print all failures.
            failed += 1
            print(f"FAIL {name}: {exc!r}")
    if failed:
        raise SystemExit(f"{failed}/{len(tests)} smoke tests failed")
    print(f"All {len(tests)} smoke tests passed")


if __name__ == "__main__":
    _run_all()
