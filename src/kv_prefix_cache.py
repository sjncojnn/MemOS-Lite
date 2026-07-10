"""Skeleton mô phỏng tái sử dụng KV-cache cho tri thức ổn định/tần suất cao.

QUAN TRỌNG (đọc trước khi hiện thực): việc tái sử dụng KV-cache *thật sự* để giảm độ trễ
(giảm time-to-first-token) phụ thuộc vào khả năng của backend suy luận, ví dụ:
- llama.cpp: `--prompt-cache <file>` lưu/khôi phục KV-cache của một prefix cố định.
- vLLM / các server hỗ trợ PagedAttention/prefix caching.
- Ollama: hỗ trợ context caching ở mức nhất định qua `context` field trong API (TODO kiểm tra
  tài liệu Ollama hiện hành trước khi cam kết behavior).

Ở mức MVP này, module chỉ theo dõi *khái niệm* — memory nào được đánh dấu 'hot' (xem
memory_ops.is_hot) thì được coi là ứng viên cho việc "ghim" prefix, và ta ghi log/thống kê
số lần logic này lẽ ra sẽ được dùng. KHÔNG có tái sử dụng KV-cache thật trong module này.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KVPrefixCacheStats:
    """Thống kê đơn giản để đo hiệu quả (giả định) của việc ghim prefix cho memory hot."""

    pin_requests: int = 0
    pin_hits: int = 0
    pinned_memory_ids: list[str] = field(default_factory=list)


class KVPrefixCache:
    """Skeleton quản lý tập các "prefix đã ghim" cho tri thức ổn định.

    TODO (nếu triển khai thật với llama.cpp):
    - `pin(memory_id, content)`: build prompt prefix cố định chứa content, chạy 1 lần qua
      backend với `--prompt-cache-file` để backend lưu KV-cache ra đĩa.
    - `build_prompt(query, pinned_ids)`: ghép các prefix đã ghim (theo đúng thứ tự đã cache)
      + phần câu hỏi động, rồi gọi lại backend với cùng `--prompt-cache-file` để tái sử dụng.
    - Cần đo latency thực tế (TTFT) có/không có prefix cache để báo cáo theo Mục 5 (Problem
      Statement) — xem scripts/eval_compare.py.
    """

    def __init__(self, max_pinned: int = 20):
        self.max_pinned = max_pinned
        self.stats = KVPrefixCacheStats()

    def pin(self, memory_id: str, content: str) -> bool:
        """Đánh dấu một memory là 'đã ghim' (giả lập). Trả về True nếu ghim thành công.

        TODO: thực sự gọi backend để build/lưu KV-cache cho `content` khi hiện thực thật.
        """
        self.stats.pin_requests += 1
        if memory_id in self.stats.pinned_memory_ids:
            return True
        if len(self.stats.pinned_memory_ids) >= self.max_pinned:
            # TODO: chọn eviction policy (LRU theo access, hoặc theo score 'hot' thấp nhất)
            return False
        self.stats.pinned_memory_ids.append(memory_id)
        return True

    def is_pinned(self, memory_id: str) -> bool:
        return memory_id in self.stats.pinned_memory_ids

    def unpin(self, memory_id: str) -> None:
        if memory_id in self.stats.pinned_memory_ids:
            self.stats.pinned_memory_ids.remove(memory_id)

    def maybe_use_cache(self, memory_id: str) -> Optional[str]:
        """Trả về "cache handle" giả lập nếu memory đã ghim, ngược lại None.

        TODO: khi hiện thực thật, trả về đường dẫn/khoá tới KV-cache đã lưu để
        qa_service.py truyền cho ollama_client/llamacpp_client tái sử dụng.
        """
        if self.is_pinned(memory_id):
            self.stats.pin_hits += 1
            return f"kv-cache-placeholder:{memory_id}"
        return None
