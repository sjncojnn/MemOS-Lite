"""Simple Memory-aware hybrid retriever for MemOS-lite.

This version deliberately avoids a large weighted reranker.  It uses a small,
transparent hybrid policy:

1. vector search over Chroma when embeddings are available;
2. keyword/BM25-like search over MemoryUnit content;
3. metadata search over category, tags, source, heading, and source_ref;
4. Reciprocal Rank Fusion (RRF) to merge the ranked lists.

The MemOS-lite difference from a plain RAG baseline is not an advanced model,
but managed memory signals: lifecycle status, provenance/source type, headings,
category/tags, hot/cold tier, recency, and unresolved-conflict handling.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Optional

from src.config import Config
from src.memory_store import MemoryStore
from src.schemas import (
    ConflictResolution,
    MemoryStatus,
    MemoryTier,
    MemoryUnit,
    RetrievedMemory,
    SourceType,
    now_utc,
)

_TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
_RAW_STOPWORDS = {
    "và", "là", "của", "cho", "thì", "mà", "nào", "như", "thế", "này", "đó", "có", "không",
    "khách", "hàng", "tôi", "anh", "chị", "em", "hay", "hoặc", "trong", "ngoài", "với", "khi",
    "nếu", "được", "cần", "phải", "gì", "sao", "ra", "về", "một", "các", "những", "toi",
    "the", "and", "or", "a", "an", "to", "of", "in", "is", "are", "what", "how",
}


@dataclass
class _Candidate:
    memory: MemoryUnit
    vector_rank: Optional[int] = None
    keyword_rank: Optional[int] = None
    metadata_rank: Optional[int] = None
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    metadata_score: float = 0.0
    rrf_score: float = 0.0
    final_score: float = 0.0


class MemoryRetriever:
    """Retrieve managed MemoryUnit objects for question answering."""

    def __init__(
        self,
        store: MemoryStore,
        config: Optional[Config] = None,
        embed_fn: Optional[Callable[[str], list[float]]] = None,
    ) -> None:
        self.store = store
        self.config = config or store.config
        if embed_fn is not None and self.store.embed_fn is None:
            self.store.embed_fn = embed_fn

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        category: Optional[str] = None,
        include_status: Optional[list[MemoryStatus]] = None,
        include_cold: bool = True,
        min_score: Optional[float] = None,
        touch: bool = True,
        category_is_strict: bool = False,
    ) -> list[RetrievedMemory]:
        """Return relevant memories for a query.

        category is a soft boost by default. Use category_is_strict=True only
        when the UI/user explicitly restricts the search scope.
        """

        query = query.strip()
        if not query:
            return []

        top_k = top_k or self.config.top_k
        statuses = include_status or [MemoryStatus.ACTIVE]
        allowed_status = {s.value if hasattr(s, "value") else str(s) for s in statuses}

        # Keep lifecycle semantics: expired memories should not be silently used.
        self.store.expire_due()

        candidate_k = max(top_k * int(getattr(self.config, "retrieval_candidate_multiplier", 4)), 20)
        strict_category = category if category_is_strict else None
        soft_category = None if category_is_strict else category

        pool = self._candidate_pool(
            statuses=statuses,
            allowed_status=allowed_status,
            category=strict_category,
            include_cold=include_cold,
        )
        if not pool:
            return []

        conflict_ids = self._unresolved_contradiction_memory_ids()
        if bool(getattr(self.config, "retrieval_exclude_conflicts", True)) and conflict_ids:
            pool = [u for u in pool if u.id not in conflict_ids]
            if not pool:
                return []

        q_tokens = _token_list(query)
        q_set = set(q_tokens)
        candidates: dict[str, _Candidate] = {u.id: _Candidate(memory=u) for u in pool}

        vector_ranked = self._vector_ranked(query, candidate_k, statuses, allowed_status, None)
        for rank, item in enumerate(vector_ranked, start=1):
            if item.memory.id not in candidates:
                # Vector metadata can be stale; hydrate and lifecycle-filter through store.search,
                # then only accept ids that satisfy this call's pool constraints.
                continue
            cand = candidates[item.memory.id]
            cand.vector_rank = rank
            cand.semantic_score = max(cand.semantic_score, item.score)

        # keyword_scores = self._keyword_scores(q_tokens, pool)
        metadata_scores = {u.id: self._metadata_score(q_set, u, None) for u in pool}

        # keyword_ranked = self._ranked_ids(keyword_scores, candidate_k)
        metadata_ranked = self._ranked_ids(metadata_scores, candidate_k)

        # for rank, memory_id in enumerate(keyword_ranked, start=1):
        #     candidates[memory_id].keyword_rank = rank
        #     candidates[memory_id].keyword_score = keyword_scores[memory_id]
        for rank, memory_id in enumerate(metadata_ranked, start=1):
            candidates[memory_id].metadata_rank = rank
            candidates[memory_id].metadata_score = metadata_scores[memory_id]

        rank_lists = [
            ([r.memory.id for r in vector_ranked if r.memory.id in candidates], 1.00),
            # (keyword_ranked, 1.00),
            (metadata_ranked, 0.70)        
        ]
        fused = _reciprocal_rank_fusion(rank_lists, k=int(getattr(self.config, "retrieval_rrf_k", 60)))
        max_rrf = max(fused.values(), default=0.0) or 1.0

        scored: list[_Candidate] = []
        threshold = self.config.min_retrieval_score if min_score is None else min_score
        for memory_id, rrf_score in fused.items():
            cand = candidates.get(memory_id)
            if cand is None:
                continue
            cand.rrf_score = rrf_score
            cand.final_score = self._final_score(cand, normalized_rrf=rrf_score / max_rrf, category=soft_category)
            if cand.final_score >= threshold:
                scored.append(cand)

        scored.sort(key=lambda c: c.final_score, reverse=True)
        scored = self._diversify(scored, top_k=top_k)
        return self._to_results(scored[:top_k], touch=touch)
    
    @staticmethod
    def _parse_memory_ref(memory_ref: str) -> tuple[str, Optional[int]]:
        """Parse memory ref dạng raw-id hoặc raw-id::v2::hxxxx."""

        parts = str(memory_ref).split("::")
        raw_id = parts[0].strip()

        expected_version: Optional[int] = None

        for part in parts[1:]:
            if part.startswith("v") and part[1:].isdigit():
                expected_version = int(part[1:])
                break

        return raw_id, expected_version


    def retrieve_by_ids(
        self,
        memory_refs: Sequence[str],
        *,
        touch: bool = False,
    ) -> list[RetrievedMemory]:
        """Load active memories from raw or version-aware cache references."""

        results: list[RetrievedMemory] = []

        for memory_ref in memory_refs:
            raw_id, expected_version = self._parse_memory_ref(str(memory_ref))

            if not raw_id:
                continue

            unit = self.store.get(raw_id)

            if unit is None:
                continue

            if unit.status != MemoryStatus.ACTIVE:
                continue

            # Cache cũ không có version vẫn được hỗ trợ.
            # Cache mới có version sẽ bị vô hiệu nếu memory đã được cập nhật.
            if expected_version is not None and unit.version != expected_version:
                continue

            results.append(
                RetrievedMemory(
                    memory=unit,
                    score=1.0,
                    rank=len(results) + 1,
                )
            )

        if touch:
            for result in results:
                self.store.touch_access(result.memory.id)

        return results

    def build_context(self, results: list[RetrievedMemory], max_chars: Optional[int] = None) -> str:
        """Format retrieved memories as compact prompt context."""

        max_chars = max_chars or self.config.max_context_chars
        blocks: list[str] = []
        used = 0

        for item in results:
            mem = item.memory
            source_ref = mem.provenance.source_ref if mem.provenance else ""
            source_type = mem.provenance.source_type.value if mem.provenance else ""
            header = (
                f"[Memory {item.rank} | score={item.score:.3f} | "
                f"source={mem.source} | "
                f"category={mem.category} | ref={source_ref}"
            )
            content = self._compact_content(mem)
            block = f"{header}\n{content}"
            remaining = max_chars - used
            if remaining <= 0:
                break
            if len(block) > remaining:
                block = block[:remaining].rstrip()
            blocks.append(block)
            used += len(block) + 2

        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Candidate generation / scoring
    # ------------------------------------------------------------------

    def _candidate_pool(
        self,
        *,
        statuses: list[MemoryStatus],
        allowed_status: set[str],
        category: Optional[str],
        include_cold: bool,
    ) -> list[MemoryUnit]:
        limit = int(getattr(self.config, "retrieval_lexical_pool_limit", 2000))
        include_expired = any((s.value if hasattr(s, "value") else str(s)) == MemoryStatus.EXPIRED.value for s in statuses)
        pool = self.store.list(
            status=statuses[0] if len(statuses) == 1 else None,
            category=category,
            include_expired=include_expired,
            limit=limit,
        )
        return [
            u for u in pool
            if u.status.value in allowed_status
            and u.is_available
            and (include_cold or u.tier != MemoryTier.COLD)
        ]

    def _vector_ranked(
        self,
        query: str,
        candidate_k: int,
        statuses: list[MemoryStatus],
        allowed_status: set[str],
        category: Optional[str],
    ) -> list[RetrievedMemory]:
        if self.store.embed_fn is None:
            return []
        semantic_min = float(getattr(self.config, "retrieval_semantic_candidate_min_score", 0.0))
        try:
            if len(statuses) == 1:
                return self.store.search(
                    query,
                    top_k=candidate_k,
                    status=statuses[0],
                    category=category,
                    min_score=semantic_min,
                    touch=False,
                )
            return [
                r for r in self.store.search(
                    query,
                    top_k=candidate_k,
                    status=None,
                    category=category,
                    min_score=semantic_min,
                    touch=False,
                )
                if r.memory.status.value in allowed_status
            ]
        except Exception:
            # If Chroma/Ollama is unavailable during a quick local demo, keyword
            # retrieval should still work instead of returning no memories.
            return []

    def _keyword_scores(self, query_tokens: list[str], pool: list[MemoryUnit]) -> dict[str, float]:
        if not query_tokens:
            return {}

        doc_tokens = {u.id: _token_list(self._search_text(u)) for u in pool}
        n_docs = max(1, len(pool))
        avg_len = sum(len(toks) for toks in doc_tokens.values()) / n_docs or 1.0

        df: Counter[str] = Counter()
        for toks in doc_tokens.values():
            df.update(set(toks))

        q_terms = set(query_tokens)
        scores: dict[str, float] = {}
        for unit in pool:
            toks = doc_tokens[unit.id]
            if not toks:
                continue
            tf = Counter(toks)
            score = 0.0
            for term in q_terms:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                idf = math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
                dl = len(toks)
                k1 = 1.5
                b = 0.75
                score += idf * (freq * (k1 + 1.0)) / (freq + k1 * (1.0 - b + b * dl / avg_len))

            # Query coverage protects short business queries from being buried in long chunks.
            coverage = len(q_terms & set(toks)) / max(1, len(q_terms))
            if coverage > 0:
                score += 0.25 * coverage
            if score > 0:
                scores[unit.id] = score
        return scores

    def _metadata_score(self, query_tokens: set[str], unit: MemoryUnit, category: Optional[str]) -> float:
        meta_tokens = set(_token_list(self._metadata_text(unit)))
        if not query_tokens or not meta_tokens:
            score = 0.0
        else:
            score = len(query_tokens & meta_tokens) / len(query_tokens)
        if category and _norm(category) == _norm(unit.category):
            score = max(score, 0.85)
        return score


    def _final_score(self, cand: _Candidate, *, normalized_rrf: float, category: Optional[str]) -> float:
        score = normalized_rrf
        unit = cand.memory
        if unit.tier == MemoryTier.HOT:
            score += 0.04
        elif unit.tier == MemoryTier.WARM:
            score += 0.015
        if category and _norm(category) == _norm(unit.category):
            score += 0.03
        return round(max(0.0, min(1.0, score)), 4)

    @staticmethod
    def _recency_score(unit: MemoryUnit) -> float:
        reference = unit.updated_at or unit.created_at
        days = max(0.0, (now_utc() - reference).total_seconds() / 86400.0)
        return 1.0 / (1.0 + days / 30.0)

    def _unresolved_contradiction_memory_ids(self) -> set[str]:
        conflicts = self.store.list_conflicts(resolution=ConflictResolution.UNRESOLVED)
        ids: set[str] = set()

        for conflict in conflicts:
            if conflict.conflict_type.value != "contradiction":
                continue
            ids.add(conflict.memory_id_a)
            ids.add(conflict.memory_id_b)

        return ids

    @staticmethod
    def _ranked_ids(scores: dict[str, float], top_k: int) -> list[str]:
        return [
            memory_id for memory_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
            if score > 0
        ][:top_k]

    def _to_results(self, ranked: list[_Candidate], *, touch: bool) -> list[RetrievedMemory]:
        results: list[RetrievedMemory] = []
        for rank, cand in enumerate(ranked, start=1):
            unit = cand.memory
            if touch:
                self.store.touch_access(unit.id)
                refreshed = self.store.get(unit.id)
                if refreshed is not None:
                    unit = refreshed
            results.append(RetrievedMemory(memory=unit, score=cand.final_score or cand.semantic_score or 1.0, rank=rank))
        return results

    @staticmethod
    def _diversify(scored: list[_Candidate], *, top_k: int) -> list[_Candidate]:
        selected: list[_Candidate] = []
        per_source: dict[str, int] = {}
        for cand in scored:
            source = cand.memory.source or "unknown"
            if per_source.get(source, 0) >= 2 and len(selected) < top_k - 1:
                continue
            selected.append(cand)
            per_source[source] = per_source.get(source, 0) + 1
            if len(selected) >= top_k:
                break
        if len(selected) < top_k:
            seen = {c.memory.id for c in selected}
            selected.extend(c for c in scored if c.memory.id not in seen)
        return selected[:top_k]

    @staticmethod
    def _compact_content(mem: MemoryUnit) -> str:
        question = str(mem.extra_metadata.get("question", "")).strip()
        answer = str(mem.extra_metadata.get("answer", "")).strip()
        if question and answer:
            return f"Q: {question}\nA: {answer}"
        heading = str(mem.extra_metadata.get("doc_heading", "")).strip()
        if heading and not mem.content.strip().startswith(heading):
            return f"{heading}\n\n{mem.content.strip()}"
        return mem.content.strip()

    @staticmethod
    def _search_text(unit: MemoryUnit) -> str:
        return " ".join(
            [
                unit.content,
                unit.summary,
                str(unit.extra_metadata.get("question", "")),
                str(unit.extra_metadata.get("answer", "")),
                str(unit.extra_metadata.get("doc_heading", "")),
                unit.category,
                " ".join(str(t) for t in unit.tags),
            ]
        )

    @staticmethod
    def _metadata_text(unit: MemoryUnit) -> str:
        return " ".join(
            [
                unit.category,
                " ".join(str(t) for t in unit.tags),
                unit.source,
                unit.provenance.source_ref if unit.provenance else "",
                unit.provenance.source_path if unit.provenance else "",
                str(unit.extra_metadata.get("doc_heading", "")),
            ]
        )


# ---------------------------------------------------------------------------
# Text/rank helpers
# ---------------------------------------------------------------------------


def _norm(text: str) -> str:
    text = str(text or "").lower().strip().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^\wÀ-ỹ]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


_STOPWORDS = {_norm(word) for word in _RAW_STOPWORDS}


def _token_list(text: str) -> list[str]:
    normalized = _norm(text)
    return [tok for tok in _TOKEN_RE.findall(normalized) if len(tok) >= 2 and tok not in _STOPWORDS]


def _reciprocal_rank_fusion(rank_lists: list[tuple[list[str], float]], *, k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ids, weight in rank_lists:
        for rank, memory_id in enumerate(ids, start=1):
            scores[memory_id] = scores.get(memory_id, 0.0) + weight / (k + rank)
    return scores
