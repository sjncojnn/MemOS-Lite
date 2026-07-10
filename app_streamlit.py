"""Streamlit demo for MemOS-lite vs a fair plain RAG baseline.

Run from the project root:
    streamlit run app_streamlit.py

What this demo is designed to show quickly:
1. Upload DOCX/XLSX while the app is running.
2. Ingest the same cleaned units into MemOS-lite and the baseline.
3. Ask one question and compare answer, latency, cache hit, and retrieved context.
4. Inspect MemOS-specific features: provenance, lifecycle status, tier, conflicts, TTL, cache.

The baseline inside this file is intentionally minimal and fair: it uses the same
LLM, same embedding model, same top-k, and the same uploaded units/chunks, but it
stores them as plain vector chunks without lifecycle/provenance-aware scheduling,
conflict handling, tiering, or QA cache.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

# Let the file work when placed at the project root.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import db
from src.client_factory import get_llm_client, reset_client_cache
from src.config import Config, load_config
from src.ingest import load_file_as_units
from src.memory_ops import add_many, add_memory, expire_due_memories, update_tiers
from src.memory_store import MemoryStore
from src.qa_cache import QACache
from src.retriever import MemoryRetriever
from src.schemas import (
    ConflictResolution,
    MemoryStatus,
    MemoryTier,
    MemoryUnit,
    Provenance,
    SourceType,
    now_utc,
)
from src.vector_store import VectorStore


SYSTEM_PROMPT = """Bạn là trợ lý hỏi đáp nghiệp vụ kênh bán.
Chỉ trả lời dựa trên CONTEXT được cung cấp.
Nếu CONTEXT không đủ thông tin, hãy nói: "Tôi chưa tìm thấy thông tin này trong tài liệu đã nạp."
Trả lời ngắn gọn, rõ ràng, đúng nghiệp vụ. Không bịa thêm chính sách, phí, hạn mức hoặc điều kiện.
"""

USER_PROMPT_TEMPLATE = """CONTEXT:
{context}

CÂU HỎI:
{question}

YÊU CẦU:
- Trả lời bằng tiếng Việt.
- Nếu có nhiều nguồn mâu thuẫn, hãy nói rõ là dữ liệu đang mâu thuẫn và cần kiểm tra lại.
- Không nhắc tới id nội bộ trừ khi người dùng hỏi.
"""


@dataclass
class BaselineHit:
    chunk_id: str
    document: str
    metadata: dict[str, Any]
    score: float
    rank: int


class PlainRAGBaseline:
    """Tiny baseline used only for the demo UI.

    It deliberately has no SQLite metadata, no lifecycle status, no conflict
    records, no tiering, and no QA cache. This keeps the comparison easy to
    explain: same knowledge and model, different memory-management layer.
    """

    def __init__(self, config: Config, embed_fn):
        self.config = config
        self.embed_fn = embed_fn
        self.vector_store = VectorStore(config.baseline_chroma_path, config.baseline_collection)

    @staticmethod
    def _baseline_id(unit: MemoryUnit) -> str:
        source_ref = unit.provenance.source_ref if unit.provenance else ""
        raw = f"{unit.source}|{source_ref}|{unit.content_hash}"
        return "base-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def ingest_units(self, units: list[MemoryUnit]) -> int:
        if not units:
            return 0
        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for unit in units:
            source_ref = unit.provenance.source_ref if unit.provenance else ""
            source_type = unit.provenance.source_type.value if unit.provenance else ""
            ids.append(self._baseline_id(unit))
            documents.append(unit.content)
            embeddings.append([float(x) for x in self.embed_fn(unit.content)])
            metadatas.append(
                {
                    "source": unit.source,
                    "source_ref": source_ref,
                    "source_type": source_type,
                    "category": unit.category,
                }
            )

        self.vector_store.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        return len(ids)

    def search(self, query: str, top_k: int) -> list[BaselineHit]:
        hits = self.vector_store.query([float(x) for x in self.embed_fn(query)], top_k=top_k)
        results: list[BaselineHit] = []
        for idx, hit in enumerate(hits, start=1):
            distance = hit.get("distance")
            try:
                score = max(0.0, min(1.0, 1.0 - float(distance))) if distance is not None else 0.0
            except (TypeError, ValueError):
                score = 0.0
            results.append(
                BaselineHit(
                    chunk_id=str(hit.get("id", "")),
                    document=str(hit.get("document", "")),
                    metadata=dict(hit.get("metadata") or {}),
                    score=round(score, 4),
                    rank=idx,
                )
            )
        return results

    def build_context(self, hits: list[BaselineHit], max_chars: int) -> str:
        blocks: list[str] = []
        used = 0
        for hit in hits:
            meta = hit.metadata
            header = (
                f"[Chunk {hit.rank}] score={hit.score:.3f} | "
                f"source_type={meta.get('source_type', '')} | source={meta.get('source', '')} | "
                f"category={meta.get('category', '')} | ref={meta.get('source_ref', '')}"
            )
            block = f"{header}\n{hit.document}"
            remaining = max_chars - used
            if remaining <= 0:
                break
            if len(block) > remaining:
                block = block[:remaining].rstrip()
            blocks.append(block)
            used += len(block) + 2
        return "\n\n".join(blocks)

    def count(self) -> int:
        return self.vector_store.count()


def memory_ref(unit: MemoryUnit) -> str:
    """Version/hash-aware ref for QA cache invalidation."""

    return f"{unit.id}::v{unit.version}::h{unit.content_hash[:12]}"


def make_config_from_sidebar() -> Config:
    config = load_config()

    st.sidebar.header("Cấu hình")
    config.default_ttl_seconds = st.sidebar.number_input(
        "TTL seconds cho memory mới, 0 = không TTL",
        min_value=0,
        value=int(config.default_ttl_seconds),
        step=60,
    )
    st.sidebar.caption(f"Runtime data: `{config.home_dir}`")
    return config


def init_runtime(config: Config):
    client = get_llm_client(config)
    embed_fn = client.embed
    store = MemoryStore(config, embed_fn=embed_fn)
    retriever = MemoryRetriever(store, config=config)
    qa_cache = QACache(config, embed_fn=embed_fn)
    baseline = PlainRAGBaseline(config, embed_fn=embed_fn)
    return client, store, retriever, qa_cache, baseline


def health_badge(client) -> bool:
    try:
        ok = client.health_check()
    except Exception:
        ok = False
    if ok:
        st.sidebar.success("Ollama: đang chạy")
    else:
        st.sidebar.error("Ollama chưa sẵn sàng")
        st.sidebar.caption("Chạy: ollama serve; ollama pull <model>; ollama pull <embedding_model>")
    return ok


def generate_answer(client, question: str, context: str, llm_ready: bool) -> str:
    if not context.strip():
        return "Tôi chưa tìm thấy thông tin này trong tài liệu đã nạp."
    if not llm_ready:
        return "LLM chưa sẵn sàng. App chỉ hiển thị retrieval context bên dưới để bạn kiểm tra trước."

    prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question)
    return client.generate(
        prompt,
        system=SYSTEM_PROMPT,
        temperature=0.0,
        top_p=0.9,
        num_predict=512,
    ).strip()


def ask_memos(
    *,
    client,
    retriever: MemoryRetriever,
    store: MemoryStore,
    qa_cache: QACache,
    question: str,
    top_k: int,
    category: Optional[str],
    bypass_cache: bool,
    llm_ready: bool,
):
    start = time.perf_counter()
    cache_hit = False
    cache_kind = "none"

    if not bypass_cache:
        exact_entry = qa_cache.get_exact(question, store=store)
        if exact_entry is not None:
            results = retriever.retrieve_by_ids(exact_entry.retrieved_memory_ids, touch=False)
            context = retriever.build_context(results)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {
                "answer": exact_entry.answer,
                "latency_ms": elapsed_ms,
                "cache_hit": True,
                "cache_kind": "exact",
                "retrieved": results,
                "context": context,
            }

    results = retriever.retrieve(
        question,
        top_k=top_k,
        category=category or None,
        category_is_strict=False,
        touch=True,
    )
    context = retriever.build_context(results)
    refs = [memory_ref(r.memory) for r in results]

    if not bypass_cache:
        semantic_entry = qa_cache.get_semantic_answer(
            question,
            store=store,
            current_memory_ids=refs,
            top_k=5,
        )
        if semantic_entry is not None:
            cache_hit = True
            cache_kind = "semantic"
            answer = semantic_entry.answer
        else:
            answer = generate_answer(client, question, context, llm_ready)
            qa_cache.set(question, answer, refs)
    else:
        answer = generate_answer(client, question, context, llm_ready)

    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "answer": answer,
        "latency_ms": elapsed_ms,
        "cache_hit": cache_hit,
        "cache_kind": cache_kind,
        "retrieved": results,
        "context": context,
    }


def ask_baseline(
    *,
    client,
    baseline: PlainRAGBaseline,
    question: str,
    top_k: int,
    max_context_chars: int,
    llm_ready: bool,
):
    start = time.perf_counter()
    hits = baseline.search(question, top_k=top_k)
    context = baseline.build_context(hits, max_chars=max_context_chars)
    answer = generate_answer(client, question, context, llm_ready)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "answer": answer,
        "latency_ms": elapsed_ms,
        "retrieved": hits,
        "context": context,
    }


def render_memos_results(results) -> None:
    if not results:
        st.info("Không có memory nào được retrieve.")
        return
    for item in results:
        mem = item.memory
        prov = mem.provenance
        source_type = prov.source_type.value if prov else ""
        source_ref = prov.source_ref if prov else ""
        with st.expander(
            f"#{item.rank} score={item.score:.3f} | tier={mem.tier.value} | status={mem.status.value} | {mem.source} | {source_ref}",
            expanded=item.rank <= 2,
        ):
            st.caption(
                f"id={mem.id} | version={mem.version} | access_count={mem.access_count} | "
                f"source_type={source_type} | category={mem.category}"
            )
            st.text(mem.content[:4000])


def render_baseline_results(hits: list[BaselineHit]) -> None:
    if not hits:
        st.info("Không có chunk nào được retrieve.")
        return
    for hit in hits:
        meta = hit.metadata
        with st.expander(
            f"#{hit.rank} score={hit.score:.3f} | {meta.get('source', '')} | {meta.get('source_ref', '')}",
            expanded=hit.rank <= 2,
        ):
            st.caption(
                f"chunk_id={hit.chunk_id} | source_type={meta.get('source_type', '')} | category={meta.get('category', '')}"
            )
            st.text(hit.document[:4000])


def memory_stats(store: MemoryStore, baseline: PlainRAGBaseline) -> dict[str, int]:
    active = store.list(status=MemoryStatus.ACTIVE, include_expired=True, limit=100_000)
    archived = store.list(status=MemoryStatus.ARCHIVED, include_expired=True, limit=100_000)
    expired = store.list(status=MemoryStatus.EXPIRED, include_expired=True, limit=100_000)
    conflicts = store.list_conflicts(limit=100_000)
    unresolved = store.list_conflicts(resolution=ConflictResolution.UNRESOLVED, limit=100_000)
    return {
        "memos_active": len(active),
        "memos_archived": len(archived),
        "memos_expired": len(expired),
        "tier_hot": sum(1 for u in active if u.tier == MemoryTier.HOT),
        "tier_warm": sum(1 for u in active if u.tier == MemoryTier.WARM),
        "tier_cold": sum(1 for u in active if u.tier == MemoryTier.COLD),
        "conflicts_total": len(conflicts),
        "conflicts_unresolved": len(unresolved),
        "baseline_chunks": baseline.count(),
    }


def render_stats_cards(store: MemoryStore, baseline: PlainRAGBaseline) -> None:
    stats = memory_stats(store, baseline)
    cols = st.columns(5)
    cols[0].metric("MemOS active", stats["memos_active"])
    cols[1].metric("Baseline chunks", stats["baseline_chunks"])
    cols[2].metric("Hot/Warm/Cold", f"{stats['tier_hot']}/{stats['tier_warm']}/{stats['tier_cold']}")
    cols[3].metric("Expired", stats["memos_expired"])
    cols[4].metric("Unresolved conflicts", stats["conflicts_unresolved"])


def _preview(text: str, limit: int = 180) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def _memory_label(unit: Optional[MemoryUnit]) -> str:
    if unit is None:
        return "missing"
    return f"status={unit.status.value} | tier={unit.tier.value} | created={unit.created_at} | source={unit.source}"


def conflict_table(store: MemoryStore, conflicts) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for c in conflicts:
        a = store.get(c.memory_id_a)
        b = store.get(c.memory_id_b)
        rows.append(
            {
                "id": c.id,
                "type": c.conflict_type.value,
                "resolution": c.resolution.value,
                "A_status": a.status.value if a else "missing",
                "A_created": a.created_at if a else "",
                "A_source": a.source if a else "",
                "A_preview": _preview(a.content if a else ""),
                "B_status": b.status.value if b else "missing",
                "B_created": b.created_at if b else "",
                "B_source": b.source if b else "",
                "B_preview": _preview(b.content if b else ""),
                "note": c.note,
                "detected_at": c.detected_at,
            }
        )
    return pd.DataFrame(rows)


def mark_conflict(config: Config, conflict_id: int, resolution: ConflictResolution, note: str) -> None:
    with db.connect(config.sqlite_path) as conn:
        db.resolve_conflict(conn, conflict_id, resolution, note=note)


def keep_by_time(config: Config, store: MemoryStore, conflict, *, newer: bool) -> None:
    a = store.get(conflict.memory_id_a)
    b = store.get(conflict.memory_id_b)
    if a is None or b is None:
        raise RuntimeError("Không tìm thấy đủ 2 memory của conflict record này.")

    # Tie-break by updated_at/id so the action is deterministic even when two rows
    # have the same created_at timestamp.
    key_a = (a.created_at, a.updated_at, a.id)
    key_b = (b.created_at, b.updated_at, b.id)
    if newer:
        keep, drop = (a, b) if key_a >= key_b else (b, a)
        action = "newer"
    else:
        keep, drop = (a, b) if key_a <= key_b else (b, a)
        action = "older"

    store.set_status(keep.id, MemoryStatus.ACTIVE, reason=f"admin_resolve_conflict_{conflict.id}_keep_{action}")
    store.set_status(drop.id, MemoryStatus.ARCHIVED, reason=f"admin_resolve_conflict_{conflict.id}_archive_other")
    resolution = ConflictResolution.KEPT_A if keep.id == conflict.memory_id_a else ConflictResolution.KEPT_B
    mark_conflict(config, conflict.id, resolution, f"admin_keep_{action}; keep={keep.id}; archived={drop.id}")


def tab_resolve_conflicts(config: Config, store: MemoryStore) -> None:
    st.subheader("4) Resolve conflicts / duplicates")
    st.caption("Đọc trực tiếp từ SQLite runtime đang dùng trong sidebar. Mặc định chỉ hiển thị unresolved records.")

    mode = st.radio("Hiển thị", ["unresolved", "all"], horizontal=True)
    resolution = ConflictResolution.UNRESOLVED if mode == "unresolved" else None
    conflicts = store.list_conflicts(resolution=resolution, limit=500)
    if not conflicts:
        st.info("Chưa có conflict/duplicate record phù hợp.")
        return

    st.dataframe(conflict_table(store, conflicts), use_container_width=True, hide_index=True)

    labels = [f"#{c.id} | {c.conflict_type.value} | {c.resolution.value} | {c.note[:70]}" for c in conflicts]
    selected = st.selectbox("Chọn record để xử lý", range(len(conflicts)), format_func=lambda i: labels[i])
    c = conflicts[selected]
    a = store.get(c.memory_id_a)
    b = store.get(c.memory_id_b)

    st.markdown("#### Candidate memories")
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Memory A", expanded=False):
            st.caption(f"id={a.id if a else c.memory_id_a} | {_memory_label(a)}")
            st.text_area("Content A", value=(a.content[:5000] if a else "missing"), height=260, disabled=True, label_visibility="collapsed")
    with col2:
        with st.expander("Memory B", expanded=False):
            st.caption(f"id={b.id if b else c.memory_id_b} | {_memory_label(b)}")
            st.text_area("Content B", value=(b.content[:5000] if b else "missing"), height=260, disabled=True, label_visibility="collapsed")

    disabled = c.resolution != ConflictResolution.UNRESOLVED
    if disabled:
        st.info(f"Record này đã được resolve: {c.resolution.value}.")

    x, y, z = st.columns(3)
    try:
        if x.button("Giữ bản cũ", disabled=disabled, use_container_width=True):
            keep_by_time(config, store, c, newer=False)
            st.success("Đã giữ bản cũ và archive bản còn lại.")
            st.rerun()
        if y.button("Giữ bản mới", disabled=disabled, use_container_width=True):
            keep_by_time(config, store, c, newer=True)
            st.success("Đã giữ bản mới và archive bản còn lại.")
            st.rerun()
        if z.button("Giữ cả hai / accept", disabled=disabled, use_container_width=True):
            mark_conflict(config, c.id, ConflictResolution.IGNORED, "admin_accept_keep_current_status")
            st.success("Đã accept conflict và giữ nguyên trạng thái hiện tại.")
            st.rerun()
    except Exception as exc:
        st.error(str(exc))


def ingest_uploaded_files(config: Config, store: MemoryStore, baseline: PlainRAGBaseline, uploaded_files, category: str):
    upload_dir = config.home_dir / "streamlit_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    total_loaded = 0
    total_baseline = 0
    memos_stats = {"inserted": 0, "exact_duplicate_skipped": 0, "conflicts": 0}

    for uploaded in uploaded_files:
        safe_name = Path(uploaded.name).name
        target_path = upload_dir / safe_name
        target_path.write_bytes(uploaded.getbuffer())

        units = load_file_as_units(target_path, config=config, category=category or "uncategorized")
        total_loaded += len(units)

        stats = add_many(store, units)
        for key in memos_stats:
            memos_stats[key] += int(stats.get(key, 0))

        # Same cleaned chunks go to plain RAG baseline.
        total_baseline += baseline.ingest_units(units)

    return {
        "loaded_units": total_loaded,
        "baseline_upserted_chunks": total_baseline,
        **memos_stats,
    }


def tab_ingest(config: Config, store: MemoryStore, baseline: PlainRAGBaseline) -> None:
    st.subheader("1) Nạp file giữa chừng vào cả MemOS-lite và baseline")
    st.caption("Cùng một file được nạp vào cả hai nhánh để so sánh công bằng về chất lượng trả lời.")

    uploaded_files = st.file_uploader(
        "Upload .docx hoặc .xlsx/.xlsm",
        type=["docx", "xlsx", "xlsm"],
        accept_multiple_files=True,
    )
    category = st.text_input("Category mặc định", value="uncategorized")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        ingest_clicked = st.button("Ingest vào MemOS + baseline", type="primary", disabled=not uploaded_files)
    with col_b:
        if st.button("Chạy expire TTL"):
            n = expire_due_memories(store)
            st.success(f"Đã expire {n} memory.")
    with col_c:
        if st.button("Cập nhật hot/warm/cold"):
            stats = update_tiers(store)
            st.success(f"Tier stats: {stats}")

    if ingest_clicked and uploaded_files:
        with st.spinner("Đang ingest và embedding..."):
            stats = ingest_uploaded_files(config, store, baseline, uploaded_files, category)
        st.success("Ingest xong.")
        st.json(stats)

    st.divider()
    render_stats_cards(store, baseline)


def tab_compare(config: Config, client, store: MemoryStore, retriever: MemoryRetriever, qa_cache: QACache, baseline: PlainRAGBaseline, llm_ready: bool) -> None:
    st.subheader("2) So sánh câu trả lời, retrieval, latency")

    examples = [
        "ViettelPay Pro là gì?",
        "Hệ thống ViettelPay Pro có những công cụ truy cập nào?",
        "Khách hàng thanh toán hóa đơn điện như thế nào?",
        "Điều kiện sử dụng dịch vụ là gì?",
    ]
    question = st.text_area("Câu hỏi", value=examples[0], height=80)
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        bypass_cache = st.checkbox("Bypass MemOS QA cache", value=False)
    with col_opt2:
        category = st.text_input("Soft category filter", value="")
    with col_opt3:
        run = st.button("Hỏi cả 2 hệ thống", type="primary")

    st.caption("Gợi ý câu hỏi nhanh: " + " | ".join(f"`{x}`" for x in examples))

    if not run:
        return
    if not question.strip():
        st.warning("Nhập câu hỏi trước.")
        return

    with st.spinner("Đang retrieve/generate..."):
        memos = ask_memos(
            client=client,
            retriever=retriever,
            store=store,
            qa_cache=qa_cache,
            question=question.strip(),
            top_k=config.top_k,
            category=category.strip() or None,
            bypass_cache=bypass_cache,
            llm_ready=llm_ready,
        )
        base = ask_baseline(
            client=client,
            baseline=baseline,
            question=question.strip(),
            top_k=config.top_k,
            max_context_chars=config.max_context_chars,
            llm_ready=llm_ready,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### MemOS-lite")
        m1, m2, m3 = st.columns(3)
        m1.metric("Latency", f"{memos['latency_ms']:.0f} ms")
        m2.metric("Cache", "hit" if memos["cache_hit"] else "miss")
        m3.metric("Cache kind", str(memos["cache_kind"]))
        st.write(memos["answer"])

    with c2:
        st.markdown("### Baseline RAG")
        b1, b2, b3 = st.columns(3)
        b1.metric("Latency", f"{base['latency_ms']:.0f} ms")
        b2.metric("Cache", "none")
        b3.metric("Retrieved", len(base["retrieved"]))
        st.write(base["answer"])

    st.divider()
    t1, t2, t3, t4 = st.tabs(["MemOS retrieved memories", "Baseline retrieved chunks", "MemOS prompt context", "Baseline prompt context"])
    with t1:
        render_memos_results(memos["retrieved"])
    with t2:
        render_baseline_results(base["retrieved"])
    with t3:
        st.text(memos["context"][:20000])
    with t4:
        st.text(base["context"][:20000])


def tab_memory_ops(store: MemoryStore, qa_cache: QACache) -> None:
    st.subheader("3) Demo khả năng riêng của MemOS-lite")

    st.markdown("#### Thêm FAQ thủ công để test update/conflict/cache")
    c1, c2 = st.columns(2)
    with c1:
        q = st.text_input("Question", value="Phí chuyển tiền là bao nhiêu?")
    with c2:
        category = st.text_input("Category", value="faq/manual")
    answer = st.text_area("Answer", value="Phí chuyển tiền là 0 đồng trong chương trình khuyến mại.", height=80)

    if st.button("Add manual FAQ vào MemOS", type="primary"):
        unit = MemoryUnit(
            content=f"Q: {q.strip()}\nA: {answer.strip()}",
            source="streamlit_manual",
            category=category.strip() or "faq/manual",
            tags=["faq", "manual"],
            tier=MemoryTier.HOT,
            provenance=Provenance(
                source_type=SourceType.FAQ,
                source_path="streamlit_manual",
                source_ref=f"manual:{now_utc().isoformat()}",
            ),
            extra_metadata={"question": q.strip(), "answer": answer.strip()},
        )
        stored, conflict_ids = add_memory(store, unit)
        if stored is None:
            st.warning("Memory bị skip vì exact duplicate.")
        else:
            st.success(f"Đã thêm memory id={stored.id}. conflict_ids={conflict_ids}")

    st.divider()
    st.markdown("#### Update / archive memory")
    memory_id = st.text_input("Memory id để update/archive", value="")
    new_content = st.text_area("Nội dung mới nếu update", value="", height=100)
    u1, u2, u3 = st.columns(3)
    with u1:
        if st.button("Update content"):
            if not memory_id.strip() or not new_content.strip():
                st.warning("Cần memory id và nội dung mới.")
            else:
                try:
                    updated = store.update_content(memory_id.strip(), new_content.strip(), summary="streamlit_update")
                    st.success(f"Updated version={updated.version}")
                except Exception as exc:
                    st.error(str(exc))
    with u2:
        if st.button("Archive memory"):
            if not memory_id.strip():
                st.warning("Cần memory id.")
            else:
                try:
                    store.delete(memory_id.strip(), hard=False)
                    st.success("Archived.")
                except Exception as exc:
                    st.error(str(exc))
    with u3:
        if st.button("Cleanup expired QA cache"):
            n = qa_cache.cleanup_expired()
            st.success(f"Deleted {n} expired QA cache rows.")

    st.divider()
    st.markdown("#### Memory list")
    status_filter = st.selectbox("Status", ["active", "archived", "expired", "all"], index=0)
    status = None if status_filter == "all" else MemoryStatus(status_filter)
    units = store.list(status=status, include_expired=True, limit=200)
    rows = []
    for u in units:
        prov = u.provenance
        rows.append(
            {
                "id": u.id,
                "status": u.status.value,
                "tier": u.tier.value,
                "version": u.version,
                "access_count": u.access_count,
                "category": u.category,
                "source_type": prov.source_type.value if prov else "",
                "source": u.source,
                "ref": prov.source_ref if prov else "",
                "content_preview": u.content[:160].replace("\n", " "),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### Conflict records")
    conflicts = store.list_conflicts(limit=200)
    conflict_rows = [
        {
            "id": c.id,
            "type": c.conflict_type.value,
            "resolution": c.resolution.value,
            "memory_a": c.memory_id_a,
            "memory_b": c.memory_id_b,
            "note": c.note,
            "detected_at": c.detected_at,
        }
        for c in conflicts
    ]
    st.dataframe(pd.DataFrame(conflict_rows), use_container_width=True, hide_index=True)


def tab_fairness_notes() -> None:
    st.subheader("4) Logic demo và cách nói khi báo cáo")
    st.markdown(
        """
**Thiết kế công bằng:**

- Cùng một file upload được parse/clean một lần rồi nạp vào cả hai nhánh.
- Cùng embedding model, cùng LLM, cùng `top_k`, cùng prompt trả lời.
- Baseline chỉ làm: `chunk -> vector search -> prompt LLM`.
- MemOS-lite làm thêm: `MemoryUnit -> provenance -> lifecycle -> tier -> conflict check -> retrieval scheduling -> QA cache`.

**Khi nhìn kết quả:**

- Nếu chất lượng câu trả lời gần ngang baseline: bình thường, vì hai hệ thống dùng cùng LLM và cùng dữ liệu.
- Điểm khác biệt cần demo là MemOS-lite có quản lý tri thức: thêm file giữa chừng, skip duplicate, phát hiện conflict, TTL expire, hot/warm/cold, cache hit, truy vết nguồn.
- Latency của MemOS-lite chỉ nên kỳ vọng giảm rõ ở câu hỏi lặp/gần lặp khi cache hit. Với câu hỏi mới, latency có thể ngang hoặc chậm hơn baseline do có thêm tầng kiểm tra memory.
"""
    )


def main() -> None:
    st.set_page_config(page_title="MemOS-lite Demo", page_icon="🧠", layout="wide")
    st.title("🧠 MemOS-lite demo: so sánh với RAG baseline")

    config = make_config_from_sidebar()

    if st.sidebar.button("Reset toàn bộ demo data", type="secondary"):
        reset_client_cache()
        if config.home_dir.exists():
            shutil.rmtree(config.home_dir)
        st.sidebar.success("Đã reset. Bấm rerun nếu app chưa tự refresh.")
        st.rerun()

    try:
        client, store, retriever, qa_cache, baseline = init_runtime(config)
    except Exception as exc:
        st.error(f"Không khởi tạo được runtime: {exc}")
        st.stop()

    llm_ready = health_badge(client)
    render_stats_cards(store, baseline)

    t_ingest, t_compare, t_ops, t_admin = st.tabs(
        ["Ingest file", "QA compare", "Memory ops", "Resolve conflicts"]
    )
    with t_ingest:
        tab_ingest(config, store, baseline)
    with t_compare:
        tab_compare(config, client, store, retriever, qa_cache, baseline, llm_ready)
    with t_ops:
        tab_memory_ops(store, qa_cache)
    with t_admin:
        tab_resolve_conflicts(config, store)


if __name__ == "__main__":
    main()
