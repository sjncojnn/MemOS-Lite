"""Pipeline hỏi đáp cho RAG baseline: vector search -> prompt -> LLM.

Baseline này KHÔNG cache, KHÔNG lifecycle, KHÔNG provenance/version/conflict. Mục đích
là làm đối chứng thuần túy cho MemOS-lite trong eval_compare.py.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from src.schemas import QAResult

try:  # Khi chạy trong project package: baseline/baseline_qa.py
    from baseline.baseline_store import BaselineVectorStore
except ImportError:  # Khi test riêng file phẳng.
    from baseline_store import BaselineVectorStore  # type: ignore

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


class BaselineQAService:
    """QA pipeline RAG thuần túy — mỗi câu hỏi luôn đi full pipeline."""

    def __init__(
        self,
        store: BaselineVectorStore,
        embed_fn: Callable[[str], list[float]],
        generate_fn: Callable[[str], str],
        top_k: int = 5,
        max_context_chars: int = 8000,
    ):
        self.store = store
        self.embed_fn = embed_fn
        self.generate_fn = generate_fn
        self.top_k = top_k
        self.max_context_chars = max_context_chars

    def _build_context(self, hits: list[dict]) -> str:
        blocks: list[str] = []
        used = 0
        for idx, hit in enumerate(hits, start=1):
            metadata = hit.get("metadata") or {}
            source = metadata.get("source", "")
            score = hit.get("score")
            score_text = f" | score={score:.3f}" if isinstance(score, (int, float)) else ""
            header = f"[Chunk {idx}] source={source}{score_text}"
            document = str(hit.get("document") or "").strip()
            if not document:
                continue
            block = f"{header}\n{document}"
            remaining = self.max_context_chars - used
            if remaining <= 0:
                break
            if len(block) > remaining:
                block = block[:remaining].rstrip()
            blocks.append(block)
            used += len(block) + 2
        return "\n\n".join(blocks)

    @staticmethod
    def _unique_sources(hits: list[dict]) -> list[str]:
        sources: list[str] = []
        seen: set[str] = set()
        for hit in hits:
            source = str((hit.get("metadata") or {}).get("source", ""))
            if source and source not in seen:
                sources.append(source)
                seen.add(source)
        return sources

    def answer(self, query: str, *, top_k: Optional[int] = None) -> QAResult:
        """Trả lời một câu hỏi bằng RAG baseline.

        Luồng chạy luôn là: embed query -> Chroma query -> build prompt -> gọi LLM.
        Không đọc/ghi QA cache để giữ baseline công bằng khi so sánh latency.
        """

        query = (query or "").strip()
        start = time.perf_counter()

        if not query:
            latency_ms = (time.perf_counter() - start) * 1000
            return QAResult(
                query=query,
                answer="Vui lòng nhập câu hỏi.",
                branch="baseline",
                latency_ms=latency_ms,
                retrieved_sources=[],
                retrieved_memory_ids=[],
                cache_hit=False,
                raw_context="",
            )

        query_embedding = self.embed_fn(query)
        hits = self.store.query(query_embedding, top_k=top_k or self.top_k)
        context = self._build_context(hits)

        if context:
            prompt = PROMPT_TEMPLATE.format(context=context, query=query)
            answer = self.generate_fn(prompt).strip()
        else:
            answer = "Chưa tìm thấy thông tin liên quan trong tài liệu."

        latency_ms = (time.perf_counter() - start) * 1000
        return QAResult(
            query=query,
            answer=answer,
            branch="baseline",
            latency_ms=latency_ms,
            retrieved_sources=self._unique_sources(hits),
            retrieved_memory_ids=[str(h.get("id", "")) for h in hits if h.get("id")],
            cache_hit=False,
            raw_context=context,
        )
