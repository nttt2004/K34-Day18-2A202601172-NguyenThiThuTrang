# Group Report — Lab 18: Production RAG

**Nhóm:** Cá nhân (làm một mình toàn bộ M1-M5)
**Ngày:** 2026-08-19

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Nguyễn Thị Thu Trang | M1: Chunking | ☑ | 13/13 |
| Nguyễn Thị Thu Trang | M2: Hybrid Search | ☑ | 5/5 |
| Nguyễn Thị Thu Trang | M3: Reranking | ☑ | 5/5 |
| Nguyễn Thị Thu Trang | M4: Evaluation | ☑ | 4/4 |
| Nguyễn Thị Thu Trang | M5: Enrichment | ☑ | 10/10 |

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.8333 | 0.8750 | +0.0417 |
| Answer Relevancy | 0.6036 | 0.5832 | -0.0204 |
| Context Precision | 0.6667 | 0.6167 | -0.0500 |
| Context Recall | 0.6500 | 0.6667 | +0.0167 |

(Nguồn: `reports/naive_baseline_report.json` vs `reports/ragas_report.json`, 20 câu hỏi test set.)

## Key Findings

1. **Biggest improvement:** Faithfulness tăng nhẹ (+0.0417) nhờ M5 enrichment (contextual prepend "Trích từ X.md...") giúp LLM bám sát nguồn hơn khi trả lời, và prompt "chỉ dựa trên context" trong `run_query` (pipeline.py) hạn chế bịa đặt so với baseline không có ràng buộc này.
2. **Biggest challenge:** Context Precision giảm (-0.05) so với naive. Nguyên nhân chính: hybrid search (BM25 + Dense + RRF) đôi khi kéo vào nhiều chunk trùng từ khóa bề mặt nhưng sai ngữ cảnh (VD case #5 trong failure_analysis: "nghỉ không lương" match nhầm case thử việc thay vì bảng ngưỡng ngày phép), và reranker (M3) không đủ mạnh để lọc bỏ các chunk nhiễu này — cross-encoder có thể đang fallback về lexical scoring khi model không load được cục bộ.
3. **Surprise finding:** Việc thêm hybrid search + reranking (production pipeline phức tạp hơn) không cải thiện đồng đều mọi metric so với naive baseline — answer_relevancy và context_precision còn giảm nhẹ. Điều này cho thấy retrieval phức tạp hơn không tự động tốt hơn nếu M1 (chunking) vẫn cắt dữ liệu không đúng ranh giới ngữ nghĩa (bảng markdown bị cắt giữa dòng — xem case #4 failure_analysis) — root cause nằm ở đầu pipeline (M1), không phải ở M2/M3.

## Presentation Notes (5 phút)

1. RAGAS scores (naive vs production): Faithfulness 0.833→0.875, Answer Relevancy 0.604→0.583, Context Precision 0.667→0.617, Context Recall 0.650→0.667. Cải thiện khiêm tốn, không đồng đều giữa các metric.
2. Biggest win — module nào, tại sao: M5 (Enrichment) — contextual prepend giúp faithfulness và context_recall nhích lên vì chunk tự mô tả rõ nguồn gốc/chủ đề hơn, giảm nhầm lẫn khi BM25 match từ khóa.
3. Case study — 1 failure, Error Tree walkthrough: Câu "Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?" — answer bị cắt cụt thành "Tổn" vì M1 cắt chunk ngay giữa dòng bảng markdown chứa đáp án đúng ("Tổng Giám đốc"). Context đúng, answer sai không phải do generation mà do chunk hóa hỏng dữ liệu nguồn trước khi tới LLM.
4. Next optimization nếu có thêm 1 giờ: Thêm rule chunking giữ nguyên bảng markdown làm 1 khối (không cắt giữa `|...|`), và kiểm tra xem CrossEncoder/Flashrank reranker có đang chạy thật hay fallback lexical — đây là 2 điểm rẻ nhất để cải thiện context_precision.
