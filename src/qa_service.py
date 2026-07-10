"""QA orchestration for MemOS-lite.

Pipeline: exact cache -> retrieval -> semantic cache -> prompt -> LLM -> cache.
Keep this file small: retrieval details live in retriever.py, cache safety lives in
qa_cache.py, and storage/lifecycle updates happen through MemoryStore.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Optional

from src.client_factory import LLMClient
from src.config import Config
from src.qa_cache import QACache
from src.retriever import MemoryRetriever
from src.schemas import CacheEntry, QAResult, RetrievedMemory

EMPTY_QUERY_ANSWER = "Vui lòng nhập câu hỏi."

NO_CONTEXT_ANSWER = "Chưa có thông tin trong kho tri thức để trả lời câu hỏi này."

PROMPT_TEMPLATE = """Bạn là trợ lý hỏi đáp nghiệp vụ.

Yêu cầu:
- Chỉ trả lời dựa trên NGỮ CẢNH được cung cấp.
- Chỉ nói chưa tìm thấy thông tin khi không có đoạn tài liệu nào liên quan trực tiếp tới câu hỏi.
- Không bịa quy trình, phí, điều kiện, thời hạn, hoặc chính sách.
- Trả lời ngắn gọn, đúng trọng tâm câu hỏi cần trả lời, trả lời bằng tiếng Việt.
- Trong ngữ cảnh có thể có các mục FAQ dạng "Q: ... / A: ...".
  + "Q:" là câu hỏi mẫu trong tài liệu, dùng để nhận biết mục FAQ đó có liên quan với câu hỏi người dùng hay không.
  + Không trả lời lại câu "Q:" trong FAQ như thể đó là câu hỏi chính.
  + Nếu câu hỏi người dùng có ý nghĩa tương đương hoặc gần với "Q:" trong FAQ, hãy ưu tiên dùng nội dung "A:" để trả lời.

### Ngữ cảnh:
{context}

### Câu hỏi cần trả lời:
{query}

### Trả lời:"""

class QAService:
    """Run one managed QA turn over MemOS-lite memory."""

    def __init__(
        self,
        retriever: MemoryRetriever,
        llm_client: LLMClient,
        cache: QACache,
        config: Config,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.cache = cache
        self.config = config

    def answer(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        category: Optional[str] = None,
    ) -> QAResult:
        """Answer a user question with retrieval, safe cache reuse, and provenance."""

        query = (query or "").strip()
        start = time.perf_counter()
        if not query:
            return self._result(start, query, EMPTY_QUERY_ANSWER, extra={"reason": "empty_query"})

        cached = self._get_exact_cache(query)
        if cached is not None:
            return self._result(
                start,
                query,
                cached.answer,
                cache_hit=True,
                retrieved_memory_ids=self._raw_ids(cached.retrieved_memory_ids),
                extra={"cache_type": "exact"},
            )

        retrieved = self.retriever.retrieve(query, top_k=top_k, category=category)
        memory_refs = self._memory_refs(retrieved)

        cached = self._get_semantic_cache(query, memory_refs)
        if cached is not None:
            return self._result(
                start,
                query,
                cached.answer,
                cache_hit=True,
                retrieved=retrieved,
                retrieved_memory_ids=self._raw_ids(cached.retrieved_memory_ids),
                raw_context=self._build_context(retrieved),
                extra={"cache_type": "semantic"},
            )

        context = self._build_context(retrieved)
        if not context:
            return self._result(start, query, NO_CONTEXT_ANSWER, extra={"reason": "no_retrieval"})

        prompt = PROMPT_TEMPLATE.format(context=context, query=query)
        answer = self.llm_client.generate(prompt).strip()
        self._set_cache(query, answer, memory_refs) 

        return self._result(
            start,
            query,
            answer,
            retrieved=retrieved,
            retrieved_memory_ids=self._raw_ids(memory_refs),
            raw_context=context,
        )

    def _build_context(self, retrieved: list[RetrievedMemory]) -> str:
        if hasattr(self.retriever, "build_context"):
            return self.retriever.build_context(retrieved, max_chars=self.config.max_context_chars)
        return "\n\n".join(f"[Nguồn: {r.memory.source}]\n{r.memory.content}" for r in retrieved)

    @staticmethod
    def _memory_refs(retrieved: Sequence[RetrievedMemory]) -> list[str]:
        """Version-aware refs used by QACache to avoid stale answers after updates."""

        refs: list[str] = []
        for item in retrieved:
            mem = item.memory
            refs.append(f"{mem.id}::v{mem.version}::h{mem.content_hash[:12]}")
        return refs

    @staticmethod
    def _raw_ids(memory_refs: Sequence[str]) -> list[str]:
        return list(dict.fromkeys(str(ref).split("::", 1)[0] for ref in memory_refs if ref))

    def _get_exact_cache(self, query: str) -> Optional[CacheEntry]:
        try:
            return self.cache.get(query, store=self.retriever.store)
        except Exception:
            return None

    def _get_semantic_cache(self, query: str, memory_refs: Sequence[str]) -> Optional[CacheEntry]:
        try:
            return self.cache.get_semantic_answer(
                query,
                store=self.retriever.store,
                current_memory_ids=memory_refs,
                top_k=max(3, self.config.top_k),
            )
        except Exception:
            return None

    def _set_cache(self, query: str, answer: str, memory_refs: Sequence[str]) -> None:
        try:
            self.cache.set(query, answer, retrieved_memory_ids=memory_refs)
        except Exception:
            # Cache is an optimization; QA correctness should not depend on it.
            return

    @staticmethod
    def _result(
        start: float,
        query: str,
        answer: str,
        *,
        cache_hit: bool = False,
        retrieved: Optional[list[RetrievedMemory]] = None,
        retrieved_memory_ids: Optional[list[str]] = None,
        raw_context: str = "",
        extra: Optional[dict] = None,
    ) -> QAResult:
        retrieved = retrieved or []
        return QAResult(
            query=query,
            answer=answer,
            branch="memos",
            latency_ms=(time.perf_counter() - start) * 1000,
            retrieved_sources=[r.memory.source for r in retrieved],
            retrieved_memory_ids=retrieved_memory_ids or [r.memory.id for r in retrieved],
            cache_hit=cache_hit,
            raw_context=raw_context,
            extra=extra or {},
        )
