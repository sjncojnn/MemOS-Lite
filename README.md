# MemOS-Lite

MemOS-Lite là MVP quản lý bộ nhớ tri thức cho hệ thống hỏi đáp nghiệp vụ tiếng Việt, được xây dựng cho bài toán hỗ trợ khách hàng/kênh bán ViettelPay Pro. Dự án lấy cảm hứng từ MemOS nhưng được thu gọn theo tinh thần đơn giản, dễ chạy và dễ kiểm chứng trên máy macOS không có GPU.

Dự án gồm hai nhánh độc lập để so sánh:

| Nhánh | Pipeline | Khả năng chính |
| --- | --- | --- |
| **RAG baseline** | chunk → embedding → vector search → prompt → LLM | RAG thuần, không cache, lifecycle, provenance hoặc conflict management |
| **MemOS-lite** | MemoryUnit → SQLite + Chroma → memory-aware retrieval → QA cache → LLM | Quản lý provenance, version, TTL, tier, duplicate, contradiction và cache |

> Mục tiêu của MemOS-Lite không phải tạo ra một mô hình sinh câu trả lời khác baseline. Hai nhánh có thể dùng cùng LLM, embedding model và dữ liệu. Khác biệt chính nằm ở lớp quản lý tri thức và khả năng tái sử dụng kết quả an toàn.

## 1. Chức năng chính

### Ingestion

- Đọc tài liệu nghiệp vụ `.docx`.
- Tách DOCX theo heading/section; bảng được lưu thành các đơn vị riêng.
- Đọc FAQ `.xlsx` hoặc `.xlsm`; mỗi dòng hỏi–đáp trở thành một `MemoryUnit`.
- Giữ nguyên tiếng Việt có dấu và chuẩn hóa khoảng trắng/ký tự điều khiển.
- Gắn provenance gồm loại nguồn, đường dẫn file và vị trí trong tài liệu.

### Memory management

- Exact duplicate được phát hiện bằng `content_hash` và bỏ qua khi ingest.
- Near-duplicate được giữ lại nhưng ghi nhận trong bảng conflict.
- FAQ có cùng câu hỏi nhưng câu trả lời khác nhau được ghi nhận là `CONTRADICTION`.
- Nội dung cập nhật được tăng `version`, tạo hash mới và re-index vector.
- Lifecycle tách biệt với serving tier:
  - `ACTIVE`, `ARCHIVED`, `EXPIRED` là trạng thái vòng đời.
  - `HOT`, `WARM`, `COLD` là mức ưu tiên phục vụ.
- Hỗ trợ TTL, access count, lifecycle log và xử lý conflict thủ công.

### Retrieval và QA cache

- Vector retrieval qua Chroma.
- Metadata ranking dựa trên category, tags, source, heading và source reference.
- Kết hợp thứ hạng bằng Reciprocal Rank Fusion (RRF).
- Category là soft boost mặc định; chỉ trở thành hard filter khi được yêu cầu rõ ràng.
- Loại bỏ memory đang nằm trong unresolved contradiction khỏi context QA.
- Exact cache theo normalized query hash.
- Semantic cache dựa trên cosine similarity, keyword coverage và các safety guard.
- Cache reference có dạng `memory_id::version::hash-prefix`, giúp vô hiệu cache khi memory bị sửa, archive hoặc expire.

> Trong phiên bản hiện tại, hàm keyword/BM25 của retriever vẫn tồn tại nhưng chưa được đưa vào rank fusion chính. Retrieval thực tế đang dùng vector ranking kết hợp metadata ranking.

## 2. Kiến trúc tổng quát

```text
DOCX / XLSX
    │
    ▼
Ingest + clean + split
    │
    ▼
MemoryUnit
    │
    ├── exact duplicate detection
    ├── near-duplicate / contradiction detection
    └── provenance + lifecycle metadata
    │
    ├───────────────┐
    ▼               ▼
SQLite           Chroma
source of truth  vector index
    │               │
    └───────┬───────┘
            ▼
Memory-aware retrieval
            │
            ▼
Exact cache → Semantic cache → LLM
            │
            ▼
QAResult + source + latency + cache information
```

SQLite là nguồn dữ liệu chính cho nội dung và trạng thái nghiệp vụ. Chroma chỉ đóng vai trò vector index và có thể được xây dựng lại từ SQLite.

## 3. Cấu trúc repository

```text
MemOS-Lite/
├── app_streamlit.py          # Demo ingest, QA compare và memory administration
├── baseline/
│   ├── baseline_ingest.py    # Chunk và ingest cho RAG baseline
│   ├── baseline_store.py     # Chroma collection riêng của baseline
│   └── baseline_qa.py        # Vector search → prompt → LLM
├── evaluation/
│   ├── eval.py               # Đánh giá answer quality và retrieval quality
│   ├── eval_qa_cache.py      # Đánh giá cache policy và latency
│   ├── dataset/              # Tạo sau khi giải nén datasets_eval.zip
│   └── runs/                 # Kết quả evaluation được sinh tự động
├── src/
│   ├── config.py             # Cấu hình và environment-variable overrides
│   ├── schemas.py            # MemoryUnit, Provenance, lifecycle, QAResult, CacheEntry
│   ├── ingest.py             # DOCX/XLSX → MemoryUnit
│   ├── db.py                 # SQLite schema và CRUD
│   ├── vector_store.py       # Chroma wrapper
│   ├── memory_store.py       # Facade kết hợp SQLite + Chroma
│   ├── memory_ops.py         # Dedup, conflict, TTL và tier rules
│   ├── retriever.py          # Retrieval, metadata ranking và RRF
│   ├── qa_cache.py           # Exact cache + semantic cache
│   ├── qa_service.py         # Exact cache → retrieve → semantic cache → LLM
│   ├── scheduler.py          # TTL, tier update và cache cleanup
│   ├── memory_manager.py     # API facade add/find/update/answer
│   ├── ollama_client.py      # Backend mặc định
│   ├── llamacpp_client.py    # Backend llama.cpp server tùy chọn
│   ├── client_factory.py     # Chọn backend
│   └── kv_prefix_cache.py    # Mô phỏng khái niệm prefix/KV cache
├── requirements.txt
└── README.md
```

## 4. Yêu cầu môi trường

- Python **3.10 trở lên**.
- macOS, Linux hoặc Windows có thể chạy Python và Chroma.
- Ollama được khuyến nghị cho LLM và embedding local.
- Không bắt buộc GPU; tốc độ sinh câu trả lời phụ thuộc model và phần cứng.

Ollama server là một ứng dụng bên ngoài Python. Cài package trong `requirements.txt` không thay thế cho việc cài và chạy Ollama server.

## 5. Cài đặt dự án

```bash
git clone https://github.com/sjncojnn/MemOS-Lite.git
cd MemOS-Lite

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Trên Windows PowerShell, kích hoạt virtual environment bằng:

```powershell
.venv\Scripts\Activate.ps1
```

## 6. Chuẩn bị Ollama

Cài Ollama từ trang chính thức hoặc bằng Homebrew trên macOS:

```bash
brew install ollama
```

Khởi động server và tải model:

```bash
ollama serve
```

Mở terminal khác trong khi server vẫn chạy:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Kiểm tra nhanh:

```bash
ollama list
```

Mặc định dự án sử dụng:

```text
LLM host        : http://localhost:11434
Generation model: llama3.2:3b
Embedding model : nomic-embed-text
```

## 7. Chuẩn bị bộ dữ liệu evaluation

File `datasets_eval.zip` được cung cấp riêng và không nằm sẵn trong repository. Sau khi tải file về, giải nén toàn bộ các file bên trong vào thư mục:

```text
evaluation/dataset/
```

Từ thư mục gốc của repository, chạy:

```bash
mkdir -p evaluation/dataset
unzip /duong/dan/toi/datasets_eval.zip -d evaluation/dataset
```

Ví dụ khi file nằm trong `Downloads` trên macOS:

```bash
mkdir -p evaluation/dataset
unzip ~/Downloads/datasets_eval.zip -d evaluation/dataset
```

Sau khi giải nén, kiểm tra:

```bash
ls -la evaluation/dataset
```

Các file dữ liệu nên nằm trực tiếp trong `evaluation/dataset/`. Nếu file ZIP tạo thêm một thư mục trung gian, hãy chuyển các file dữ liệu từ thư mục đó lên `evaluation/dataset/` trước khi chạy evaluation.

Cấu trúc mong đợi:

```text
evaluation/
├── dataset/
│   ├── <golden_qa_file>.xlsx
│   ├── <qa_cache_dataset>.csv
│   └── ...
├── eval.py
└── eval_qa_cache.py
```

Không cần cài thêm thư viện Python để giải nén ZIP. Có thể dùng lệnh `unzip` có sẵn trên macOS/Linux hoặc giải nén bằng Finder/Explorer.

## 8. Chạy giao diện Streamlit

Từ thư mục gốc của repository:

```bash
source .venv/bin/activate
streamlit run app_streamlit.py
```

Giao diện gồm các tab chính:

1. **Ingest file**
   - Upload `.docx`, `.xlsx` hoặc `.xlsm`.
   - File được parse một lần và nạp vào cả MemOS-Lite lẫn baseline.
   - MemOS-Lite thực hiện dedup và conflict detection.

2. **QA compare**
   - Hỏi cùng một câu cho MemOS-Lite và RAG baseline.
   - So sánh answer, latency, cache hit và retrieved context.
   - Có thể bật `Bypass MemOS QA cache` để đo cold path.

3. **Memory ops**
   - Thêm FAQ thủ công.
   - Update content và tăng version.
   - Archive memory.
   - Xem provenance, tier, lifecycle và conflict records.

4. **Resolve conflicts**
   - Giữ bản cũ.
   - Giữ bản mới.
   - Chấp nhận giữ cả hai.

Dữ liệu runtime mặc định được lưu tại:

```text
.memos_lite_data/
├── memos_lite.sqlite3
├── chroma_memos/
└── chroma_baseline/
```

Nút **Reset toàn bộ demo data** sẽ xóa thư mục runtime này. Hãy sao lưu trước nếu cần giữ dữ liệu.

## 9. Quy trình chạy từ đầu

Quy trình đề xuất cho một máy mới:

```text
1. Clone repository.
2. Tạo virtual environment và cài requirements.
3. Khởi động Ollama, tải generation model và embedding model.
4. Chạy Streamlit.
5. Upload tài liệu nghiệp vụ và FAQ trong tab Ingest file.
6. Kiểm tra số lượng MemOS memories và baseline chunks.
7. Thử QA compare để xác nhận retrieval/LLM hoạt động.
8. Giải nén datasets_eval.zip vào evaluation/dataset/.
9. Chạy quality/retrieval evaluation.
10. Chạy QA-cache evaluation riêng.
```

Evaluation sử dụng dữ liệu đã được ingest trong `.memos_lite_data`. Nếu chưa ingest hoặc trỏ nhầm `--home`, retrieval có thể trả về rỗng.

## 10. Đánh giá chất lượng câu trả lời và retrieval

`evaluation/eval.py` chạy evaluation mà **không sử dụng QA cache**. Mục tiêu là so sánh retrieval và generation giữa MemOS-Lite với baseline một cách công bằng.

Dataset golden QA hỗ trợ Excel hoặc CSV. Các cột chính:

```text
id
question
category
answerable
reference_answer
gold_source
gold_ref
gold_quote
note
```

Các cột tùy chọn:

```text
gold_memory_id
enabled
review_status
```

Ví dụ chạy cả hai hệ thống:

```bash
python evaluation/eval.py \
  --mode both \
  --dataset evaluation/dataset/golden_qa_small.xlsx \
  --home ./.memos_lite_data \
  --top-k 5
```

Nếu tên file trong ZIP khác, thay `golden_qa_small.xlsx` bằng tên file thực tế trong `evaluation/dataset/`.

Các mode hỗ trợ:

```text
--mode memos   Chỉ chạy MemOS-Lite
--mode rag     Chỉ chạy RAG baseline
--mode both    Chạy cả hai hệ thống
```

Một số tùy chọn hữu ích:

```bash
--limit 20              # Chạy thử 20 câu
--snapshot-data         # Copy data runtime trước khi eval
--strict-category       # Dùng category như hard filter
--skip-unapproved       # Chỉ chạy row đã approved
--evidence-threshold 0.55
--seed 42
```

Ví dụ chạy nhanh 20 câu:

```bash
python evaluation/eval.py \
  --mode both \
  --dataset evaluation/dataset/golden_qa_small.xlsx \
  --home ./.memos_lite_data \
  --top-k 5 \
  --limit 20
```

Các metric chính:

- `avg_token_f1`
- `avg_rouge_l`
- `hit_at_1`, `hit_at_3`, `hit_at_5`
- `mrr`
- `context_precision_at_5`
- `best_evidence_overlap`
- `answerability_accuracy`
- `avg_latency_ms`

Kết quả được tạo trong:

```text
evaluation/runs/run_YYYYMMDD_HHMMSS/
├── summary.csv
├── qa_results.csv
├── retrieval_results.csv
├── review_cases.xlsx
├── judge_input.xlsx
└── config.json
```

`judge_input.xlsx` được chuẩn bị để review hoặc đưa sang một bước LLM-judge riêng. `eval.py` hiện không tự động gọi LLM-judge lần hai.

## 11. Đánh giá QA cache

Cache evaluation được tách khỏi quality evaluation để tránh trộn hai mục tiêu:

- Policy có dự đoán đúng hit/miss hay không.
- False semantic hit có được hạn chế hay không.
- Exact cache và semantic cache giảm latency bao nhiêu.

Dataset cache là CSV với bốn cột bắt buộc:

```text
case_id,label,query_1,query_2
```

Trong đó:

- `query_1` là câu seed được chạy trước để tạo cache.
- `query_2` là câu dùng để kiểm tra semantic hit/miss.
- `label` nhận các giá trị như `hit`, `miss`, `positive`, `negative`, `1`, `0`.

Chạy evaluation bằng:

```bash
python evaluation/eval_qa_cache.py \
  --dataset evaluation/dataset/<qa_cache_dataset>.csv \
  --home ./.memos_lite_data \
  --top-k 5 \
  --cache-repeats 3
```

Thay `<qa_cache_dataset>.csv` bằng tên file thực tế sau khi giải nén ZIP.

Các metric được in ra gồm:

- Seed miss rate.
- Exact hit rate.
- Semantic policy accuracy.
- Semantic precision.
- Semantic hit recall.
- False hit rate.
- Median miss latency.
- Median exact-cache latency và speedup.
- Median semantic-cache latency và speedup.

Kết quả chi tiết được ghi vào:

```text
evaluation/runs/qa_cache/YYYYMMDD_HHMMSS/qa_cache_results.csv
```

## 12. Cấu hình runtime

Cấu hình mặc định nằm trong `src/config.py`. Có thể override trực tiếp bằng environment variables mà không cần sửa code.

| Environment variable | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `MEMOS_LITE_HOME` | `./.memos_lite_data` | Thư mục SQLite và Chroma |
| `MEMOS_LITE_LLM_BACKEND` | `ollama` | `ollama` hoặc `llamacpp` |
| `MEMOS_LITE_OLLAMA_HOST` | `http://localhost:11434` | Địa chỉ Ollama server |
| `MEMOS_LITE_OLLAMA_MODEL` | `llama3.2:3b` | Generation model |
| `MEMOS_LITE_OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `MEMOS_LITE_REQUEST_TIMEOUT_SECONDS` | `120` | HTTP timeout |
| `MEMOS_LITE_CHUNK_SIZE_TOKENS` | `1200` | Kích thước chunk gần đúng |
| `MEMOS_LITE_CHUNK_OVERLAP_TOKENS` | `150` | Chunk overlap gần đúng |
| `MEMOS_LITE_DEFAULT_TTL_SECONDS` | `0` | TTL mặc định; `0` là không expire |
| `MEMOS_LITE_TOP_K` | `5` | Số kết quả retrieval |
| `MEMOS_LITE_MAX_CONTEXT_CHARS` | `8000` | Giới hạn context gửi LLM |
| `MEMOS_LITE_NEAR_DUPLICATE_THRESHOLD` | `0.95` | Ngưỡng near-duplicate |
| `MEMOS_LITE_QA_CACHE_ENABLED` | `true` | Bật exact QA cache |
| `MEMOS_LITE_QA_CACHE_TTL_SECONDS` | `100` | TTL của QA cache |
| `MEMOS_LITE_QA_SEMANTIC_CACHE_ENABLED` | `true` | Bật semantic cache |

Ví dụ thay model và data directory trong một terminal:

```bash
export MEMOS_LITE_HOME=./.memos_lite_data
export MEMOS_LITE_OLLAMA_MODEL=llama3.2:3b
export MEMOS_LITE_OLLAMA_EMBEDDING_MODEL=nomic-embed-text
export MEMOS_LITE_TOP_K=5

streamlit run app_streamlit.py
```

Các biến được đọc trực tiếp từ môi trường. Dự án hiện không tự động nạp file `.env` trong source chính.

## 13. Chính sách memory và cache

### Duplicate và contradiction

- Exact duplicate: bỏ qua khi ingest.
- Near-duplicate: vẫn lưu và ghi conflict record.
- Contradiction: vẫn lưu để admin xử lý, nhưng mặc định không đưa các memory liên quan vào QA context khi conflict chưa được resolve.

### Lifecycle và tier

- `ACTIVE`: được phép dùng cho retrieval.
- `ARCHIVED`: vẫn còn trong SQLite nhưng không dùng để trả lời.
- `EXPIRED`: đã hết TTL và không dùng để trả lời.
- `HOT`: FAQ hoặc memory có tần suất truy cập cao/gần đây.
- `WARM`: memory active bình thường.
- `COLD`: memory ít dùng nhưng chưa bị archive/expire.

### Cache invalidation

Một cache entry chỉ hợp lệ khi toàn bộ memory reference liên quan vẫn:

- Tồn tại.
- Có trạng thái sử dụng được.
- Có đúng version.
- Có content-hash prefix phù hợp.
- Không nằm trong unresolved contradiction.

Cache là lớp tối ưu. Nếu cache gặp lỗi, QA service tiếp tục chạy retrieval và LLM thay vì làm hỏng câu trả lời.

## 14. So sánh công bằng với baseline

Trong giao diện và evaluation:

- Hai nhánh dùng cùng generation model.
- Hai nhánh dùng cùng embedding model.
- Có thể dùng cùng `top_k` và giới hạn context.
- Baseline không dùng cache.
- MemOS-Lite bổ sung quản lý memory, không thay đổi bản chất LLM.

Vì vậy, chất lượng câu trả lời của hai nhánh có thể gần nhau. Giá trị chính cần quan sát ở MemOS-Lite là:

- Lưu trữ xuyên phiên.
- Provenance và khả năng truy vết nguồn.
- Duplicate/conflict handling.
- Lifecycle và TTL.
- Hot/warm/cold tiering.
- Version-aware cache invalidation.
- Latency thấp hơn khi exact/semantic cache hit đúng.

Với câu hỏi mới hoàn toàn và cache miss, MemOS-Lite có thể ngang hoặc chậm hơn baseline do có thêm các bước kiểm tra memory và cache.

## 15. Giới hạn hiện tại

- Đây là Stage-1 MVP, không phải triển khai đầy đủ của MemOS.
- Keyword/BM25 retrieval chưa được bật trong RRF chính.
- Conflict resolution cần admin quyết định; chưa có LLM tự động phân xử.
- Semantic cache dùng heuristic và cần hiệu chỉnh threshold trên dữ liệu thật.
- `kv_prefix_cache.py` chỉ mô phỏng khái niệm; chưa tái sử dụng KV-cache thực sự.
- Chưa có multi-tenant, permission/governance chi tiết hoặc đồng bộ phân tán.
- Chưa có fine-tuning/LoRA, parameter memory hoặc activation memory thực sự.
- SQLite timestamp đang dùng naive UTC, phù hợp MVP local nhưng chưa tối ưu cho hệ thống đa vùng.

## 16. Xử lý lỗi thường gặp

### `Cannot connect to Ollama`

```bash
ollama serve
ollama list
```

Đảm bảo tên model trong `src/config.py` hoặc environment variable trùng với model đã pull.

### `streamlit: command not found`

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Không retrieve được memory

- Kiểm tra đã ingest file trong tab **Ingest file** hay chưa.
- Kiểm tra `.memos_lite_data/memos_lite.sqlite3` tồn tại.
- Kiểm tra Chroma collections không rỗng.
- Kiểm tra memory không bị `ARCHIVED`, `EXPIRED` hoặc thuộc unresolved contradiction.

### Evaluation chạy rất nhanh nhưng metric retrieval bằng 0

- Kiểm tra `--home` có trỏ đúng thư mục đã ingest hay không.
- Kiểm tra dataset có cột `question`, không phải một tên cột khác.
- Kiểm tra baseline collection đã được ingest nếu chạy `--mode rag` hoặc `--mode both`.
- Mở `qa_results.csv` và xem cột `error`.

### Chroma hoặc SQLite không đồng bộ sau nhiều lần thử

Sao lưu dữ liệu cần thiết, sau đó reset qua Streamlit hoặc xóa runtime để ingest lại:

```bash
rm -rf .memos_lite_data
```

## 17. Phạm vi ngoài MVP

Các nội dung sau chưa thuộc phạm vi hiện tại:

- Governance và permission chi tiết.
- Multi-tenant memory service.
- Memory marketplace hoặc chia sẻ memory liên mô hình.
- Multi-agent scheduling.
- Fine-tuning/LoRA để nội tại hóa tri thức.
- Production deployment, authentication, monitoring và horizontal scaling.
