from __future__ import annotations

"""
Module 3: Reranking
-------------------
Cung cấp các cơ chế đánh giá và sắp xếp lại (rerank) kết quả tìm kiếm nhằm tăng độ chính xác.
Hỗ trợ Cross-encoder (chính) và Flashrank (thay thế siêu nhẹ), kèm công cụ đo lường độ trễ.
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Any

# Đưa thư mục gốc vào PYTHONPATH để nạp config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: Dict[str, Any]
    rank: int


class CrossEncoderReranker:
    """
    Sử dụng mô hình Cross-Encoder để chấm điểm trực tiếp từng cặp (query, document).
    Nếu mô hình lỗi/không có sẵn, hệ thống tự động chuyển về thuật toán đếm từ cơ bản (lexical fallback).
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._load_attempted = False

    def _load_model(self):
        """Khởi tạo mô hình CrossEncoder (đảm bảo chỉ thử tải 1 lần duy nhất)."""
        if self._model is None and not self._load_attempted:
            self._load_attempted = True
            try:
                from sentence_transformers import CrossEncoder
                
                # Bắt buộc dùng model từ cache/local để tránh lỗi tải ngầm gây treo hệ thống
                self._model = CrossEncoder(self.model_name, local_files_only=True)
            except Exception as exc:
                print(f"CrossEncoder không khả dụng, tự động dùng Lexical Fallback. Lỗi: {exc}")
        return self._model

    def _lexical_fallback_score(self, query: str, document_text: str) -> float:
        """
        Chấm điểm dự phòng dựa trên tỉ lệ từ khóa trùng khớp (Lexical Overlap).
        Chỉ được gọi khi model AI không thể load.
        """
        query_terms = set(query.lower().split())
        if not query_terms:
            return 0.0
            
        doc_terms = document_text.lower().split()
        match_count = len(query_terms.intersection(doc_terms))
        return match_count / max(len(query_terms), 1)

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = RERANK_TOP_K) -> List[RerankResult]:
        """Tiến hành chấm điểm và chọn lọc lại top_k kết quả tốt nhất."""
        if not documents or top_k <= 0:
            return []

        model = self._load_model()
        scores: List[float] = []

        # Bước 1: Tính toán điểm số (Scores)
        if model is not None:
            # Gói thành list các tuples (query, text) để predict hàng loạt
            pairs = [(query, str(doc.get("text", ""))) for doc in documents]
            raw_scores = model.predict(pairs)
            
            # Đảm bảo kết quả luôn là list (xử lý case đầu vào chỉ có 1 document)
            scores = [raw_scores] if isinstance(raw_scores, (int, float)) else list(raw_scores)
        else:
            scores = [self._lexical_fallback_score(query, str(doc.get("text", ""))) for doc in documents]

        # Bước 2: Ghép cặp điểm với tài liệu và sắp xếp giảm dần
        scored_docs = sorted(zip(scores, documents), key=lambda x: float(x[0]), reverse=True)

        # Bước 3: Định dạng kết quả đầu ra
        return [
            RerankResult(
                text=str(doc.get("text", "")),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=dict(doc.get("metadata", {})),
                rank=rank,
            )
            for rank, (score, doc) in enumerate(scored_docs[:top_k])
        ]


class FlashrankReranker:
    """
    Phương án Rerank dự phòng siêu nhẹ (<5ms). 
    Phù hợp cho các môi trường không có GPU.
    """
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = RERANK_TOP_K) -> List[RerankResult]:
        if not documents or top_k <= 0:
            return []
            
        try:
            from flashrank import Ranker, RerankRequest

            if self._model is None:
                self._model = Ranker()
                
            # Tạo list dictionary chuẩn theo format mà Flashrank yêu cầu
            passages = [
                {"id": str(idx), "text": str(doc.get("text", ""))} 
                for idx, doc in enumerate(documents)
            ]
            
            # Chạy mô hình
            request = RerankRequest(query=query, passages=passages)
            results = self._model.rerank(request)
            
            # Trích xuất và đối chiếu kết quả (đã tách khối lệnh để dễ debug)
            final_results = []
            for rank, res in enumerate(results[:top_k]):
                original_idx = int(res["id"])
                original_doc = documents[original_idx]
                
                final_results.append(RerankResult(
                    text=res["text"],
                    original_score=float(original_doc.get("score", 0.0)),
                    rerank_score=float(res["score"]),
                    metadata=dict(original_doc.get("metadata", {})),
                    rank=rank
                ))
            return final_results

        except Exception as exc:
            print(f"Flashrank gặp sự cố. Bỏ qua reranking... Lỗi: {exc}")
            return []


def benchmark_reranker(reranker: Any, query: str, documents: List[Dict[str, Any]], n_runs: int = 5) -> Dict[str, float]:
    """Đo lường thời gian trễ (latency) của mô hình Rerank qua nhiều lần chạy."""
    times = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        reranker.rerank(query, documents)
        
        # Lưu kết quả theo đơn vị milliseconds (ms)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        times.append(elapsed_ms)
        
    return {
        "avg_ms": sum(times) / len(times), 
        "min_ms": min(times), 
        "max_ms": max(times)
    }


if __name__ == "__main__":
    # Dữ liệu giả lập để test luồng chạy
    test_query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    test_docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {"source": "hr_v2023"}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {"source": "it_policy"}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {"source": "hr_v2023"}},
    ]
    
    print("Đang khởi tạo mô hình CrossEncoderReranker...")
    tester = CrossEncoderReranker()
    
    print("\nKết quả Rerank:")
    print("-" * 65)
    for result in tester.rerank(test_query, test_docs):
        print(f"[Hạng {result.rank + 1}] Điểm mới: {result.rerank_score:.4f} | {result.text}")