# Individual Reflection — Lab 18

**Tên:** Nguyễn Thị Thu Trang
**Module phụ trách:** M1, M2, M3, M4, M5 (làm cá nhân toàn bộ pipeline)

---

## 1. Đóng góp kỹ thuật

- Module đã implement:
  - **M1 — Chunking** (`src/m1_chunking.py`): `chunk_semantic`, `chunk_hierarchical`, `chunk_structure_aware`, `compare_strategies`.
  - **M2 — Hybrid Search** (`src/m2_search.py`): `segment_vietnamese`, `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion`, `HybridSearch`.
  - **M3 — Reranking** (`src/m3_rerank.py`): `CrossEncoderReranker.rerank`, `FlashrankReranker.rerank`, `benchmark_reranker`.
  - **M4 — Evaluation** (`src/m4_eval.py`): `evaluate_ragas` (4 metric RAGAS qua Gemini OpenAI-compatible endpoint), `failure_analysis` (diagnostic tree theo worst_metric).
  - **M5 — Enrichment** (`src/m5_enrichment.py`): `summarize_chunk`, `generate_hypothesis_questions`, `contextual_prepend`, `extract_metadata`, `_enrich_single_call` (combined mode 1 call/chunk).
- Các hàm/class chính đã viết: liệt kê ở trên — tổng cộng 5 module, ghép lại trong `src/pipeline.py::build_pipeline` + `evaluate_pipeline`.
- Số tests pass: 37/37 (M1: 13, M2: 5, M3: 5, M4: 4, M5: 10)

## 2. Kiến thức học được

- Khái niệm mới nhất: Reciprocal Rank Fusion (RRF) để merge kết quả BM25 + Dense mà không cần chuẩn hóa score trực tiếp (`score = Σ 1/(k + rank)`), và kỹ thuật Parent-Child chunking (retrieve ở granularity nhỏ để tăng precision, nhưng trả về context ở granularity lớn hơn để giữ đủ ngữ cảnh).
- Điều bất ngờ nhất: Pipeline "production" (hybrid search + rerank + enrichment) không tự động thắng naive baseline trên mọi metric — context_precision và answer_relevancy còn giảm nhẹ so với baseline đơn giản. Nguyên nhân gốc rễ hóa ra nằm ở bước đầu tiên (M1 chunking cắt giữa bảng markdown), không phải ở các bước "nâng cao" hơn — cho thấy lỗi ở đầu pipeline sẽ lan ra và không thể sửa được bằng cách thêm rerank/enrichment ở cuối.
- Kết nối với bài giảng: Error Tree diagnostic (Output sai → Context đúng? → do M1 hay M2? → prompt hay retrieval?) đúng như phần lý thuyết về debug RAG theo tầng — không được kết luận "model hallucinate" ngay khi thấy `faithfulness` thấp mà phải truy ngược xem context có đủ hay không trước.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: RAGAS ban đầu cấu hình dùng OpenAI-based LLM/embeddings mặc định, không phù hợp với API key đang có sẵn (Gemini). Ngoài ra một số PDF trong `data/` là scan ảnh không có text layer (`BCTC.pdf`, `Nghi_dinh_so_13-2023...pdf`), bị `load_documents()` bỏ qua hoàn toàn → mất thông tin ngay từ bước load dữ liệu nếu không để ý cảnh báo.
- Cách giải quyết: Chuyển toàn bộ LLM/embedding calls (M4 evaluate_ragas, M5 enrichment, pipeline generation) sang Gemini qua endpoint tương thích OpenAI (`langchain_openai.ChatOpenAI`/`OpenAIEmbeddings` trỏ `base_url` sang Gemini) thay vì gọi thẳng OpenAI — giữ nguyên interface RAGAS/OpenAI SDK, chỉ đổi client config (commit "chuyển sang gemini"). Với PDF scan, chấp nhận giới hạn hệ thống text-based hiện tại (không OCR) và ghi rõ cảnh báo thay vì crash âm thầm.
- Thời gian debug: ~1-2 giờ cho phần chuyển đổi Gemini (do phải đồng bộ config ở nhiều nơi: `config.py`, `m4_eval.py`, `m5_enrichment.py`, `pipeline.py`), phần còn lại (chunking/retrieval logic) mất nhiều thời gian hơn ở việc đọc failure cases để hiểu tại sao context_precision giảm.

## 4. Nếu làm lại

- Sẽ làm khác điều gì: Ưu tiên `chunk_structure_aware` (giữ nguyên bảng markdown, section theo header) làm chiến lược mặc định cho các tài liệu có bảng (lương, thẩm quyền phê duyệt) thay vì `chunk_hierarchical` cắt theo kích thước ký tự cố định — vì nhiều failure case (case #3, #4 trong `failure_analysis.md`) đều do bảng bị cắt giữa dòng.
- Module nào muốn thử tiếp: M3 (Reranking) — muốn benchmark thật kỹ xem `CrossEncoderReranker` có đang load được model cục bộ hay luôn fallback về lexical scoring, vì đây có khả năng là nguyên nhân âm thầm khiến context_precision không cải thiện như kỳ vọng.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 3 (làm cá nhân, không có phối hợp nhóm) |
| Problem solving | 4 |
