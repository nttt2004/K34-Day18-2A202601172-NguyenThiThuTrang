# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Cá nhân
**Thành viên:** Nguyễn Thị Thu Trang → M1, M2, M3, M4, M5 (làm toàn bộ pipeline)

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8333 | 0.8750 | +0.0417 |
| Answer Relevancy | 0.6036 | 0.5832 | -0.0204 |
| Context Precision | 0.6667 | 0.6167 | -0.0500 |
| Context Recall | 0.6500 | 0.6667 | +0.0167 |

Nguồn: `reports/naive_baseline_report.json` và `reports/ragas_report.json` (20 câu hỏi test set, 10/20 câu bị flag là "failure" theo `failure_analysis()` trong M4).

---

## Bottom-5 Failures

(Xếp theo `avg_score` tăng dần từ `reports/ragas_report.json`)

### #1
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30 triệu nằm trong khoảng 5-50 triệu nên cần Giám đốc phòng ban (Director) phê duyệt. Ngoài ra, mua sắm thiết bị CNTT cần có xác nhận cấu hình kỹ thuật từ phòng CNTT trước khi đề xuất. Cần đính kèm ít nhất 3 báo giá vì trên 10 triệu.
- **Got:** "Không tìm thấy." (avg_score = 0.0)
- **Worst metric:** faithfulness (nhưng thực chất câu trả lời không hallucinate — nó từ chối trả lời)
- **Error Tree:**
  - Answer đúng ground truth? → Không, model trả lời "Không tìm thấy" trong khi ground truth có sẵn.
  - Context có chứa bằng chứng cần thiết? → **Không.** Cả 3 context trả về đều nói về "chi phí đào tạo 30 triệu" và phê duyệt nghỉ phép/đào tạo — không context nào chứa bảng thẩm quyền phê duyệt mua sắm (`mua_sam.md`).
  - Vì context thiếu → do M1 (chunk) hay M2 (retrieval)? → **M2 (retrieval).** Query dùng con số "30 triệu" + "laptop", nhưng BM25 (qua `segment_vietnamese`) khớp mạnh với "30.000.000 VNĐ/khóa" trong `hoan_chi_dao_tao.md` vì cùng chứa số "30" và từ khoá "phê duyệt", trong khi văn bản đúng (`mua_sam.md`) dùng "Từ 5.000.000 - 50.000.000 VNĐ" (không match trực tiếp "30"). Đây là lỗi vocabulary/entity mismatch giữa query và tài liệu đích, RRF không đủ mạnh để kéo `mua_sam.md` lên top-k.
- **Root cause:** M2 retrieval — hybrid search không match được truy vấn có số tiền cụ thể với bảng ngưỡng phê duyệt (dạng "khoảng giá trị"), dẫn đến chunk đúng không nằm trong context được đưa cho LLM.
- **Suggested fix:** Thêm bước "numeric range expansion" khi enrich chunk (M5) — sinh thêm câu hỏi giả định kiểu "mua thiết bị giá X thì ai duyệt?" với X là các mốc cụ thể (dùng HyQA đã có sẵn trong M5) để bridge vocabulary gap giữa số tiền cụ thể và khoảng giá trị trong bảng. Test lại: thêm câu hỏi này vào `test_set` của M4 và kiểm tra `context_recall` của riêng câu hỏi > 0.8 ở lần eval tiếp theo.

### #2
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** "Không tìm thấy." (avg_score = 0.25, worst_metric = answer_relevancy)
- **Error Tree:**
  - Answer đúng ground truth? → Không.
  - Context có chứa bằng chứng cần thiết? → **Không.** Context trả về nói về nghỉ ốm, nghỉ phép đặc biệt, nghỉ không lương — không có chunk nào về công thức tính phép theo thâm niên hay bảng lương Senior.
  - Nguyên nhân do M1 hay M2? → **M2 (retrieval)**, có khả năng phối hợp với M1: câu hỏi ghép 2 ý (số ngày phép + mức lương) nhưng hệ thống retrieve top-k chung cho toàn câu hỏi, nên cả 2 khía cạnh đều bị loãng — không chunk nào về "phép năm + thâm niên" lẫn "bảng lương Senior" lọt vào top-k vì mỗi khía cạnh chỉ được 1 phần trọng số truy vấn.
- **Root cause:** M2 retrieval không xử lý tốt câu hỏi multi-hop/multi-aspect (2 câu hỏi ghép làm 1); top_k hiện tại không đủ rộng để chứa chunk cho cả 2 khía cạnh.
- **Suggested fix:** Với câu hỏi multi-aspect, cân nhắc query decomposition (tách thành 2 sub-query rồi gộp kết quả) trước khi retrieval, hoặc tăng `HYBRID_TOP_K`/`RERANK_TOP_K` khi phát hiện câu hỏi có nhiều dấu "và". Kiểm tra bằng cách thêm case tương tự vào test set M4 và theo dõi `context_recall` tăng khi bật decomposition.

### #3
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior cao nhất là 20.000.000 VNĐ/tháng. Lương thử việc = 85% x 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** "Không tìm thấy." (avg_score = 0.25, worst_metric = answer_relevancy)
- **Error Tree:**
  - Answer đúng ground truth? → Không.
  - Context có chứa bằng chứng cần thiết? → **Một phần.** Context có công thức "85% lương thử việc" (từ `thu_viec.md` và `bang_luong_2024.md`) nhưng **không có** con số cụ thể "Junior cao nhất 20 triệu" — bảng lương chi tiết theo cấp bậc không nằm trong top-3 context.
  - Vì context thiếu chi tiết → do M1 hay M2? → **M1 (chunking).** `bang_luong_2024.md` rất có thể có bảng lương theo cấp bậc (Junior/Senior...) nằm ở section khác trong cùng file, nhưng `chunk_hierarchical`/`chunk_basic` cắt theo kích thước ký tự cố định nên bảng chi tiết mức lương Junior bị tách ra một chunk khác không được retrieve, trong khi phần "Lương thử việc = 85%" (chung, không có số) lại được ưu tiên vì match từ khoá "lương thử việc" trực tiếp với query.
- **Root cause:** M1 chunking cắt bảng lương chi tiết ra khỏi phần công thức tính lương thử việc → context bị thiếu con số cần để hoàn thành phép tính; M2 retrieval chỉ vớt được nửa thông tin (công thức, không có số liệu).
- **Suggested fix:** Với các file dạng bảng lương/tài chính, ưu tiên `chunk_structure_aware` (giữ nguyên bảng markdown) thay vì `chunk_hierarchical` cắt theo ký tự, tránh xé bảng. Test lại bằng `tests/test_m1.py`: thêm assertion rằng bảng markdown (`| ... | ... |`) không bị chunk cắt giữa dòng, và theo dõi câu hỏi này đạt answer_relevancy > 0.7 ở lần eval tiếp theo.

### #4
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** "Tổn" (avg_score = 0.3126, worst_metric = faithfulness, diagnosis: LLM hallucinating)
- **Error Tree:**
  - Answer đúng ground truth? → Không — answer bị cắt cụt giữa chừng ("Tổn" thay vì "Tổng Giám đốc").
  - Context có chứa bằng chứng cần thiết? → **Có, nhưng bị cắt.** Context #2 chứa đúng bảng thẩm quyền phê duyệt, nhưng dòng cuối bảng bị cắt ngay tại "Tổn" (do M1 chunking cắt theo `chunk_size` cố định giữa bảng markdown, đúng vị trí giá trị "Tổng Giám đốc").
  - Vì context đúng nhưng answer sai → đây là lỗi generation/prompt hay do chunk bị cắt cụt? → Answer sai **vì context đầu vào đã bị cắt cụt từ M1**, LLM chỉ copy lại đúng phần bị cắt ("Tổn") thay vì hallucinate nội dung mới — nói cách khác, RAGAS gắn nhãn "faithfulness" thấp nhưng root cause thực sự nằm ở M1 (chunk bị cắt giữa bảng), không phải do prompt.
- **Root cause:** M1 chunking cắt chunk giữa 1 dòng bảng markdown (`chunk_hierarchical`/`chunk_basic` chia theo `\n\n` và giới hạn ký tự, không nhận diện được bảng `| ... |` là 1 đơn vị không thể tách), làm mất phần cuối câu trả lời đúng.
- **Suggested fix:** Dùng `chunk_structure_aware` (đã có sẵn, parse theo header) kết hợp thêm rule "không cắt giữa 1 block bảng markdown" (regex phát hiện dòng bắt đầu bằng `|` và giữ nguyên toàn bộ bảng trong 1 chunk). Verify bằng cách thêm unit test vào `tests/test_m1.py` kiểm tra chunk chứa bảng `mua_sam.md` không bị cắt giữa dòng, và chạy lại câu hỏi này ở lần eval tiếp theo để faithfulness đạt 1.0.

### #5
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ 16-30 ngày cần phê duyệt của Giám đốc điều hành (CEO). Lưu ý: nghỉ trên 14 ngày không lương, nhân viên phải tự đóng phần bảo hiểm của mình.
- **Got:** "Trường hợp nhân viên thử việc cần nghỉ việc riêng (nghỉ không lương) thì phải được trưởng phòng phê duyệt." (avg_score = 0.3179, worst_metric = context_precision, diagnosis: Too many irrelevant chunks)
- **Error Tree:**
  - Answer đúng ground truth? → Không — answer trả lời về trường hợp **nhân viên thử việc**, không phải quy định số ngày nghỉ (16-30 ngày → CEO).
  - Context có chứa bằng chứng cần thiết? → **Không đủ.** Context #1 (`thu_viec.md`, không liên quan đến thang phê duyệt theo số ngày) được retrieve với độ ưu tiên cao hơn context thật sự chứa bảng ngưỡng ngày nghỉ theo cấp phê duyệt (không xuất hiện trong top-3 context).
  - Vì context thiếu bảng ngưỡng đúng → do M1 hay M2? → **M2 (retrieval/reranking).** Câu hỏi chứa "nghỉ không lương" + "phê duyệt", BM25 khớp tốt với `thu_viec.md` (cũng nói "nghỉ không lương... trưởng phòng phê duyệt") do trùng từ khóa bề mặt, trong khi chunk đúng (bảng ngưỡng ngày → cấp phê duyệt) có thể không dùng đúng các từ khóa này hoặc bị xếp hạng thấp hơn sau rerank.
- **Root cause:** M2 retrieval + M3 reranking ưu tiên chunk trùng từ khóa bề mặt ("nghỉ không lương", "phê duyệt") nhưng sai ngữ cảnh (case thử việc thay vì bảng ngưỡng theo số ngày), context_precision thấp vì lẫn nhiều chunk không liên quan trực tiếp đến câu hỏi.
- **Suggested fix:** Cải thiện M3 reranker — đảm bảo cross-encoder/flashrank thực sự được dùng (kiểm tra log fallback lexical) và tăng discriminative power để phân biệt "case đặc thù thử việc" với "quy tắc chung theo số ngày". Có thể thêm metadata filter (category = "nghỉ phép", không phải "thử việc") trước khi rerank. Test lại: theo dõi `context_precision` của câu hỏi này ở lần eval tiếp theo, mục tiêu > 0.6.

## Case Study (cho presentation)

**Question chọn phân tích:** #4 — "Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?"

**Error Tree walkthrough:**
1. Output đúng? → Không, answer bị cắt cụt thành "Tổn" thay vì "Tổng Giám đốc (CEO)".
2. Context đúng? → Có — context #2 đúng đoạn bảng thẩm quyền phê duyệt (`mua_sam.md`), nhưng bị cắt ngay tại giá trị cần trả lời.
3. Query rewrite OK? → Không liên quan — vấn đề không nằm ở query, mà ở việc chunk hóa dữ liệu nguồn.
4. Fix ở bước: **M1 (chunking)** — cần chunk-aware cho bảng markdown, không cắt giữa 1 dòng bảng.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Viết rule chunking nhận diện block bảng markdown (`|...|`) như 1 đơn vị nguyên tử, không tách giữa `chunk_hierarchical`/`chunk_basic`.
- Thêm 3-5 câu hỏi multi-aspect / số tiền cụ thể vào test set M4 để phát hiện sớm các lỗi vocabulary-gap tương tự #1, #2.
- Kiểm tra lại xem `FlashrankReranker`/`CrossEncoderReranker` có thực sự load được model hay đang fallback lexical (log "⚠️ ... unavailable") — vì fallback lexical yếu hơn nhiều so với cross-encoder thật, ảnh hưởng trực tiếp đến context_precision (#5).
