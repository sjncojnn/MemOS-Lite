"""QA cache for MemOS-lite.

Storage policy:
- SQLite qa_cache is the source of truth for exact query hash -> answer.
- Chroma stores cached-query embeddings for semantic candidate lookup.
- Semantic reuse requires both cached-query cosine similarity and a keyword
  score showing that the cached answer covers the new query.
- Cached memory references must still be valid and contradiction-free.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import timedelta
from typing import Optional

from src import db
from src.config import Config
from src.memory_store import MemoryStore
from src.schemas import (
    CacheEntry,
    ConflictResolution,
    ConflictType,
    compute_query_hash,
    now_utc,
)
from src.vector_store import VectorStore

EmbedFn = Callable[[str], Sequence[float]]

SENSITIVE_QUERY_RE = re.compile(
    r"\b(phí|phi|hạn mức|han muc|lãi suất|lai suat|ngày|ngay|thời hạn|thoi han|"
    r"điều kiện|dieu kien|tỷ lệ|ty le|số tiền|so tien|giá|gia|bao nhiêu|bao nhieu)\b",
    re.IGNORECASE,
)

KEYWORD_TOKEN_RE = re.compile(r"[a-z0-9]+")
RAW_TOKEN_RE = re.compile(r"[A-Za-zÀ-ỹ0-9]+", re.UNICODE)

# Common words should not make two different business questions look equivalent.
# Domain terms such as BCCS, PIN, BHXH, bảo hiểm, học phí, nạp tiền, etc. remain.
KEYWORD_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "by", "for", "from", "how", "in",
    "is", "of", "on", "or", "the", "to", "what", "when", "where", "which",
    "who", "why",
    "anh", "cac", "can", "chi", "cho", "cua", "co", "de", "do", "duoc",
    "em", "gi", "hay", "hoac", "khach", "hang", "khi", "khong", "la",
    "lam", "ma", "mot", "nao", "neu", "ngoai", "nhieu", "nhu", "nhung",
    "phai", "ra", "sao", "the", "thi", "toi", "trong", "tu", "ve", "voi",
    "app", "dung", "nguoi", "pro", "qua", "tren", "ung", "viettelpay",
}

DEFAULT_KEYWORD_SCORE_THRESHOLD = 0.65
DEFAULT_SENSITIVE_KEYWORD_SCORE_THRESHOLD = 0.75


class QACache:
    """Exact cache plus cosine-candidate and keyword-score semantic cache."""

    def __init__(self, config: Config, embed_fn: Optional[EmbedFn] = None):
        self.config = config
        self.embed_fn = embed_fn
        db.init_db(config.sqlite_path)
        self.vector_store: Optional[VectorStore] = None
        if embed_fn is not None:
            self.vector_store = VectorStore(
                config.chroma_path,
                collection_name=getattr(
                    config,
                    "qa_cache_collection",
                    "memos_lite_qa_cache",
                ),
            )

    # ------------------------------------------------------------------
    # Exact cache
    # ------------------------------------------------------------------

    def get(self, query: str, store: MemoryStore) -> Optional[CacheEntry]:
        if not self.config.qa_cache_enabled:
            return None
        return self.get_exact(query, store=store)

    def get_exact(self, query: str, store: MemoryStore) -> Optional[CacheEntry]:
        if not self.config.qa_cache_enabled:
            return None

        query_hash = compute_query_hash(query)
        with db.connect(self.config.sqlite_path) as conn:
            entry = db.get_valid_qa_cache(
                conn,
                query_hash,
                now=now_utc(),
                increment_hit=False,
            )

        if entry is None or not self._memory_refs_still_valid(entry, store):
            return None

        self._increment_hit(query_hash)
        entry.hit_count += 1
        return entry

    # ------------------------------------------------------------------
    # Semantic cache
    # ------------------------------------------------------------------

    def get_semantic_answer(
        self,
        query: str,
        *,
        store: MemoryStore,
        current_memory_ids: Sequence[str],
        top_k: int = 5,
    ) -> Optional[CacheEntry]:
        """Return a safe semantic-cache hit.

        Policy:
        1. Use cached-query cosine similarity to find candidates.
        2. Require the cached answer to pass the keyword-score threshold.
        3. Require cached memory refs to remain active and version-valid.
        4. Reject unresolved contradictions.

        ``current_memory_ids`` is retained only for QAService compatibility.
        Retrieval overlap is no longer part of semantic-cache acceptance.
        """

        _ = current_memory_ids
        if not self._semantic_enabled():
            return None

        similarity_threshold = self._answer_threshold(query)
        keyword_threshold = self._keyword_threshold(query)
        candidates = self._semantic_candidates(query, top_k=top_k)
        scores = self.keyword_scores(
            query,
            [entry for entry, _ in candidates],
        )

        for entry, similarity in candidates:
            if similarity < similarity_threshold:
                continue
            if scores.get(entry.query_hash, 0.0) < keyword_threshold:
                continue
            if not self._memory_refs_still_valid(entry, store):
                continue
            if self._has_unresolved_contradiction(
                entry.retrieved_memory_ids,
                store,
            ):
                continue

            self._increment_hit(entry.query_hash)
            entry.hit_count += 1
            return entry

        return None

    # ------------------------------------------------------------------
    # Write / cleanup
    # ------------------------------------------------------------------

    def set(
        self,
        query: str,
        answer: str,
        retrieved_memory_ids: Optional[Sequence[str]] = None,
    ) -> None:
        if not self.config.qa_cache_enabled:
            return

        ids = list(
            dict.fromkeys(
                str(value)
                for value in (retrieved_memory_ids or [])
                if value
            )
        )
        ttl = max(0, int(self.config.qa_cache_ttl_seconds))
        expires_at = now_utc() + timedelta(seconds=ttl) if ttl > 0 else None
        query_hash = compute_query_hash(query)

        with db.connect(self.config.sqlite_path) as conn:
            db.upsert_qa_cache(
                conn,
                query_hash,
                query,
                answer,
                expires_at=expires_at,
                retrieved_memory_ids=ids,
            )
            conn.execute(
                "UPDATE qa_cache SET hit_count = 0 WHERE query_hash = ?",
                (query_hash,),
            )

        if self.vector_store is not None and self.embed_fn is not None:
            self.vector_store.upsert(
                ids=[query_hash],
                embeddings=[[float(x) for x in self.embed_fn(query)]],
                documents=[query],
                metadatas=[
                    {
                        "query_hash": query_hash,
                        "expires_at": expires_at.isoformat() if expires_at else "",
                        "memory_ids": ",".join(ids),
                    }
                ],
            )

    def invalidate(self, query: str) -> None:
        query_hash = compute_query_hash(query)
        with db.connect(self.config.sqlite_path) as conn:
            conn.execute(
                "DELETE FROM qa_cache WHERE query_hash = ?",
                (query_hash,),
            )
        if self.vector_store is not None:
            self.vector_store.delete([query_hash])

    def cleanup_expired(self) -> int:
        now = now_utc()
        with db.connect(self.config.sqlite_path) as conn:
            rows = conn.execute(
                "SELECT query_hash FROM qa_cache "
                "WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now.isoformat(),),
            ).fetchall()
            expired_ids = [str(row["query_hash"]) for row in rows]
            deleted = db.delete_expired_qa_cache(conn, now=now)

        if self.vector_store is not None and expired_ids:
            self.vector_store.delete(expired_ids)
        return deleted

    # ------------------------------------------------------------------
    # Semantic candidate and thresholds
    # ------------------------------------------------------------------

    def _semantic_enabled(self) -> bool:
        return bool(
            self.config.qa_cache_enabled
            and getattr(self.config, "qa_semantic_cache_enabled", True)
            and self.embed_fn is not None
            and self.vector_store is not None
        )

    def _semantic_candidates(
        self,
        query: str,
        *,
        top_k: int,
    ) -> list[tuple[CacheEntry, float]]:
        if self.vector_store is None or self.embed_fn is None:
            return []

        hits = self.vector_store.query(
            [float(x) for x in self.embed_fn(query)],
            top_k=max(top_k * 4, 20),
            where=None,
        )

        candidates: list[tuple[CacheEntry, float]] = []
        with db.connect(self.config.sqlite_path) as conn:
            for hit in hits:
                query_hash = str(
                    hit.get("id")
                    or hit.get("metadata", {}).get("query_hash")
                    or ""
                )
                if not query_hash:
                    continue

                entry = db.get_valid_qa_cache(
                    conn,
                    query_hash,
                    now=now_utc(),
                    increment_hit=False,
                )
                if entry is None:
                    continue

                candidates.append(
                    (entry, self._distance_to_score(hit.get("distance")))
                )
                if len(candidates) >= top_k:
                    break

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates

    def _answer_threshold(self, query: str) -> float:
        if SENSITIVE_QUERY_RE.search(query):
            return float(
                getattr(
                    self.config,
                    "qa_semantic_sensitive_threshold",
                    0.96,
                )
            )
        return float(
            getattr(
                self.config,
                "qa_semantic_answer_threshold",
                0.94,
            )
        )

    def _keyword_threshold(self, query: str) -> float:
        if SENSITIVE_QUERY_RE.search(query):
            return float(
                getattr(
                    self.config,
                    "qa_semantic_sensitive_keyword_score_threshold",
                    DEFAULT_SENSITIVE_KEYWORD_SCORE_THRESHOLD,
                )
            )
        return float(
            getattr(
                self.config,
                "qa_semantic_keyword_score_threshold",
                DEFAULT_KEYWORD_SCORE_THRESHOLD,
            )
        )

    @staticmethod
    def _distance_to_score(distance: object) -> float:
        if distance is None:
            return 0.0
        try:
            return max(0.0, min(1.0, 1.0 - float(distance)))
        except (TypeError, ValueError):
            return 0.0

    # ------------------------------------------------------------------
    # Keyword scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_keyword_text(text: str) -> str:
        normalized = unicodedata.normalize(
            "NFD",
            (text or "").lower().replace("đ", "d"),
        )
        return "".join(
            char
            for char in normalized
            if unicodedata.category(char) != "Mn"
        )

    @classmethod
    def _keyword_tokens(cls, text: str) -> list[str]:
        normalized = cls._normalize_keyword_text(text)
        return [
            token
            for token in KEYWORD_TOKEN_RE.findall(normalized)
            if len(token) > 1 and token not in KEYWORD_STOPWORDS
        ]

    @classmethod
    def _critical_tokens(cls, text: str) -> set[str]:
        """Keep exact acronyms and numeric values as high-value query anchors."""

        critical: set[str] = set()
        for raw_token in RAW_TOKEN_RE.findall(text or ""):
            normalized = cls._normalize_keyword_text(raw_token)
            if not normalized or normalized in KEYWORD_STOPWORDS:
                continue
            if any(char.isdigit() for char in raw_token) or (
                raw_token.isupper() and len(raw_token) >= 2
            ):
                critical.add(normalized)
        return critical

    @classmethod
    def keyword_scores(
        cls,
        query: str,
        entries: Sequence[CacheEntry],
    ) -> dict[str, float]:
        """Score whether cached answers cover the new query.

        This follows the lightweight retriever design:
        - BM25-like term score over cached answers;
        - metadata-style query-token overlap;
        - ordered bigram overlap for short business phrases;
        - exact acronym/number anchors such as BCCS, PIN, TCTD, dates, amounts.

        The final score is normalized to [0, 1].
        """

        query_tokens = cls._keyword_tokens(query)
        query_terms = list(dict.fromkeys(query_tokens))
        if not query_terms or not entries:
            return {}

        documents = {
            entry.query_hash: cls._keyword_tokens(entry.answer)
            for entry in entries
        }
        n_docs = max(1, len(documents))
        avg_len = (
            sum(len(tokens) for tokens in documents.values()) / n_docs
        ) or 1.0

        document_frequency: Counter[str] = Counter()
        for tokens in documents.values():
            document_frequency.update(set(tokens))

        query_set = set(query_terms)
        query_bigrams = set(zip(query_tokens, query_tokens[1:]))
        critical_terms = cls._critical_tokens(query)
        scores: dict[str, float] = {}

        for entry in entries:
            answer_tokens = documents[entry.query_hash]
            if not answer_tokens:
                scores[entry.query_hash] = 0.0
                continue

            answer_set = set(answer_tokens)
            term_frequency = Counter(answer_tokens)
            answer_length = len(answer_tokens)
            k1 = 1.5
            b = 0.75

            bm25_matched = 0.0
            bm25_possible = 0.0
            for term in query_terms:
                idf = math.log(
                    1.0
                    + (n_docs - document_frequency[term] + 0.5)
                    / (document_frequency[term] + 0.5)
                )
                bm25_possible += idf

                frequency = term_frequency.get(term, 0)
                if frequency <= 0:
                    continue

                contribution = idf * (
                    frequency * (k1 + 1.0)
                ) / (
                    frequency
                    + k1 * (
                        1.0 - b + b * answer_length / avg_len
                    )
                )
                # Cap each term at its ideal contribution so the normalized
                # BM25 component stays in [0, 1].
                bm25_matched += min(idf, contribution)

            bm25_score = (
                bm25_matched / bm25_possible
                if bm25_possible > 0
                else 0.0
            )

            # Same idea as retriever._metadata_score: query-side overlap.
            metadata_score = len(query_set & answer_set) / len(query_set)

            if query_bigrams:
                answer_bigrams = set(zip(answer_tokens, answer_tokens[1:]))
                phrase_score = (
                    len(query_bigrams & answer_bigrams)
                    / len(query_bigrams)
                )
            else:
                phrase_score = metadata_score

            critical_score = (
                len(critical_terms & answer_set) / len(critical_terms)
                if critical_terms
                else metadata_score
            )

            final_score = (
                0.34 * bm25_score
                + 0.53 * metadata_score
                + 0.03 * phrase_score
                + 0.10 * critical_score
            )
            scores[entry.query_hash] = round(
                max(0.0, min(1.0, final_score)),
                6,
            )

        return scores

    # ------------------------------------------------------------------
    # Cache validity / contradiction guards
    # ------------------------------------------------------------------

    def _increment_hit(self, query_hash: str) -> None:
        with db.connect(self.config.sqlite_path) as conn:
            db.increment_qa_cache_hit(conn, query_hash)

    @staticmethod
    def _parse_memory_ref(
        memory_ref: str,
    ) -> tuple[str, Optional[int], str]:
        parts = str(memory_ref).split("::")
        raw_id = parts[0]
        version: Optional[int] = None
        hash_prefix = ""
        for part in parts[1:]:
            if len(part) >= 2 and part[0] == "v" and part[1:].isdigit():
                version = int(part[1:])
            elif len(part) >= 2 and part[0] == "h":
                hash_prefix = part[1:]
        return raw_id, version, hash_prefix

    @classmethod
    def _raw_memory_ids(cls, memory_ids: Sequence[str]) -> list[str]:
        ids = [
            cls._parse_memory_ref(str(value))[0]
            for value in memory_ids
            if value
        ]
        return list(dict.fromkeys(ids))

    @classmethod
    def _memory_refs_still_valid(
        cls,
        entry: CacheEntry,
        store: MemoryStore,
    ) -> bool:
        for memory_ref in entry.retrieved_memory_ids:
            raw_id, expected_version, hash_prefix = cls._parse_memory_ref(
                str(memory_ref)
            )
            unit = store.get(raw_id)
            if unit is None or not unit.is_available:
                return False
            if expected_version is not None and unit.version != expected_version:
                return False
            if hash_prefix and not unit.content_hash.startswith(hash_prefix):
                return False
        return True

    @classmethod
    def _has_unresolved_contradiction(
        cls,
        memory_ids: Sequence[str],
        store: MemoryStore,
    ) -> bool:
        ids = set(cls._raw_memory_ids(memory_ids))
        if not ids:
            return False

        try:
            conflicts = store.list_conflicts(
                resolution=ConflictResolution.UNRESOLVED,
                limit=10_000,
            )
        except Exception:
            # Fail closed: cache reuse is optional, correctness is not.
            return True

        return any(
            conflict.conflict_type == ConflictType.CONTRADICTION
            and (
                conflict.memory_id_a in ids
                or conflict.memory_id_b in ids
            )
            for conflict in conflicts
        )