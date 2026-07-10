# memos_lite

MVP hệ thống quản lý bộ nhớ cho LLM phục vụ hỏi-đáp nghiệp vụ kênh bán (ViettelPay Pro),
lấy cảm hứng từ [MemOS](https://github.com/MemTensor/MemOS) nhưng thu gọn tối đa theo tinh
thần "extremely simple" của [MiniRAG](https://github.com/HKUDS/MiniRAG). Chạy local trên
macOS, không cần GPU.

Dự án có **hai nhánh độc lập** để so sánh:

| Nhánh | Ý tưởng | Thư mục |
| --- | --- | --- |
| RAG baseline | chunk → embed → vector search → prompt → LLM | `baseline/` |
| MemOS-lite | MemoryUnit (content + metadata) → SQLite + Chroma → retrieve có lifecycle/provenance → QA → cập nhật usage/cache/lifecycle | `src/` |

> ⚠️ Đây là **skeleton**: chữ ký hàm/class, docstring, TODO đã có sẵn và import được,
> nhưng phần lớn logic nghiệp vụ (retrieval ranking, conflict resolution, prompt
> engineering, đánh giá LLM-judge...) còn để TODO cho nhóm hiện thực tiếp.

## Cấu trúc thư mục

```
memos_lite/
  src/                  # Lõi MemOS-lite
    config.py           # Cấu hình tập trung (đường dẫn, model, ngưỡng lifecycle)
    schemas.py           # MemoryUnit, MemoryStatus, Provenance, QAResult, ...
    ingest.py            # Đọc .docx/.xlsx -> MemoryUnit thô
    db.py                # SQLite: metadata, lifecycle_log, provenance, conflicts, qa_cache
    vector_store.py      # Wrapper Chroma cho MemOS-lite
    memory_store.py      # Kết hợp db.py + vector_store.py thành 1 interface lưu trữ
    memory_ops.py        # Duplicate/conflict detection, lifecycle rules (MVP)
    retriever.py         # Truy hồi có lọc theo lifecycle + rerank đơn giản
    scheduler.py         # Rule-based hot/cold/expired transition
    qa_cache.py          # Cache câu trả lời theo query hash
    ollama_client.py     # LLM + embedding client qua Ollama (backend mặc định)
    llamacpp_client.py   # Skeleton backend llama.cpp (chưa hiện thực sâu)
    kv_prefix_cache.py   # Skeleton mô phỏng tái sử dụng KV-cache cho tri thức ổn định
    client_factory.py    # Chọn LLM client theo config
    qa_service.py        # Pipeline hỏi đáp: retrieve -> cache -> prompt -> LLM -> update
    memory_manager.py    # Facade trung tâm: add/find/update/answer/run_scheduler
  baseline/              # Nhánh RAG baseline, độc lập, không lifecycle/cache/provenance
    baseline_ingest.py
    baseline_store.py
    baseline_qa.py
  scripts/               # CLI entrypoints
    ingest_docs.py
    ingest_faq.py
    run_qa.py
    run_scheduler.py
    eval_compare.py       # So sánh baseline vs MemOS-lite trên golden set -> CSV
  tests/                 # Smoke tests (pytest)
```

## Cài đặt

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Cài & chạy Ollama (backend LLM/embedding mặc định)
# https://ollama.com
ollama pull llama3.2:3b        # hoặc model nhỏ hơn cho máy không GPU
ollama pull nomic-embed-text
```

Sao chép `.env.example` thành `.env` (TODO) hoặc chỉnh trực tiếp `src/config.py` để đổi
model, đường dẫn dữ liệu, ngưỡng hot/cold, TTL...

## Chạy thử (MVP)

```bash
# 1. Ingest tài liệu nghiệp vụ (.docx) và FAQ (.xlsx)
python scripts/ingest_docs.py --input-dir data/docs
python scripts/ingest_faq.py --input-file data/faq.xlsx

# 2. Hỏi đáp tương tác qua MemOS-lite
python scripts/run_qa.py --branch memos

# 3. Hỏi đáp tương tác qua RAG baseline (để so sánh)
python scripts/run_qa.py --branch baseline

# 4. Chạy scheduler rule-based (hot/cold/expired)
python scripts/run_scheduler.py

# 5. So sánh 2 nhánh trên golden set, xuất report CSV
python scripts/eval_compare.py --golden-set data/golden_set.csv --out report.csv
```

## Nguyên tắc thiết kế

- **Đơn giản trước, đúng sau**: mỗi module một trách nhiệm, tránh trừu tượng hoá thừa,
  không có multi-agent, không fine-tune/LoRA, không permission chi tiết ở giai đoạn này.
- **MemoryUnit là đơn vị tri thức tối thiểu** (không phải MemCube đầy đủ như MemOS gốc):
  content + metadata đủ dùng cho lifecycle, provenance, dedup/conflict.
- **SQLite cho metadata có cấu trúc** (lifecycle, provenance, conflict, cache), **Chroma cho
  vector** — tách rõ hai mối quan tâm, dễ debug bằng cách mở trực tiếp file `.sqlite`.
- **Baseline hoàn toàn độc lập** với MemOS-lite: không import lifecycle/cache/provenance,
  để phép so sánh công bằng "trước/sau".
- **KV-cache reuse (`kv_prefix_cache.py`) chỉ là mô phỏng ở mức khái niệm** trong MVP này —
  việc tái sử dụng KV-cache thật sự phụ thuộc vào khả năng của backend suy luận (vd.
  `--prompt-cache` của llama.cpp, prefix caching của vLLM). TODO nêu rõ trong file.

## Giới hạn phạm vi (Out of scope cho MVP này)

Không hiện thực: governance/permission chi tiết, đa tenant, chợ trao đổi bộ nhớ, chia sẻ
liên mô hình, fine-tuning/LoRA để nội tại hoá tri thức vào tham số, multi-agent scheduling.
Các phần này để "Giai đoạn sau" theo Problem Statement.

## TODO tổng quát

- [ ] Viết `.env.example` và load config từ biến môi trường.
- [ ] Hiện thực logic reranking trong `retriever.py`.
- [ ] Hiện thực conflict resolution có ý nghĩa trong `memory_ops.py`.
- [ ] Kết nối `eval_compare.py` với LLM-judge thật (hiện là placeholder).
- [ ] Hiện thực `llamacpp_client.py` nếu cần chạy hoàn toàn offline không qua Ollama server.
