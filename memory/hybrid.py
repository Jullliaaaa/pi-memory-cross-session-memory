"""任務二：Hybrid retrieval（BM25 + 本地多語言 embedding）。

設計重點：
- 純 BM25 路徑（memory/bm25.py::bm25_search）維持確定性、不被污染，供測試驗證。
- 本檔提供兩種融合策略，皆透過 core.retrieve() 以環境變數 RETRIEVAL_MODE 啟用：
    * weighted：分數正規化後加權相加（保留分數大小，需調 alpha）
    * rrf     ：Reciprocal Rank Fusion，用「排名」融合（免正規化、免調權重）
- embedding 模型固定為多語言模型，才能修好跨語言題（中文查詢 ↔ 英文記憶）。

確定性 vs 機率性的分界：
- BM25 是 100% 確定性的「外殼」。
- embedding 是「機率性核心」：同一模型版本下，同一段文字輸出固定向量，
  因此排序仍可重現；但更換模型 / 套件版本可能造成向量微變。
  重現性靠 requirements.txt 鎖定模型與版本來維持。
"""
from __future__ import annotations
import os
from functools import lru_cache

from .bm25 import bm25_search, tokenize

# 多語言模型：支援中英跨語言對齊，才能接到跨語言平行記憶。
_DEFAULT_MODEL = os.environ.get(
    "EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
)


@lru_cache(maxsize=1)
def _get_model(name: str):
    """延遲載入 embedding 模型；只在真的走 hybrid 時才 import，
    確保未安裝 sentence-transformers 時不影響純 BM25 核心。"""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def _embed(texts: list[str]):
    """把一批文字轉成正規化後的向量（normalize 後內積即 cosine）。"""
    model = _get_model(_DEFAULT_MODEL)
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def _embedding_scores(query: str, docs: list[dict]):
    """回傳每篇文件對 query 的 cosine 相似度（與 docs 同序）。"""
    import numpy as np
    if not docs:
        return np.array([])
    doc_texts = [d["text"] for d in docs]
    doc_vecs = _embed(doc_texts)        # (N, dim)，已正規化
    q_vec = _embed([query])[0]          # (dim,)，已正規化
    # 已正規化，內積即 cosine
    return doc_vecs @ q_vec             # (N,)


def _bm25_scores(query: str, docs: list[dict]):
    """借用確定性 BM25 算每篇分數（與 docs 同序）。
    bm25_search 回傳前 k 筆，這裡傳 k=len(docs) 取全部，再對回原序。"""
    import numpy as np
    ranked = bm25_search(query, docs, k=len(docs))
    score_by_id = {r["id"]: r["score"] for r in ranked}
    return np.array([score_by_id.get(d["id"], 0.0) for d in docs])


def _minmax(arr):
    """把分數正規化到 [0,1]；全相同時回傳全 0，避免除以 0。"""
    import numpy as np
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _weighted_search(query, docs, k, alpha=0.5):
    """策略一：加權分數融合。
    final = alpha * BM25_norm + (1 - alpha) * embedding_norm
    alpha 越大越偏字面（BM25），越小越偏語義（embedding）。"""
    import numpy as np
    if not docs:
        return []
    bm = _minmax(_bm25_scores(query, docs))
    em = _minmax(_embedding_scores(query, docs))
    final = alpha * bm + (1 - alpha) * em
    order = np.argsort(-final, kind="stable")  # 高到低，同分維持原序
    return [{"id": docs[i]["id"], "score": float(final[i])} for i in order[:k]]


def _rank_map(scores):
    """把分數轉成排名（1 = 最高分）；同分用 stable 排序維持原序。"""
    import numpy as np
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    rank = [0] * len(scores)
    for r, idx in enumerate(order, start=1):
        rank[idx] = r
    return rank


def _rrf_search(query, docs, k, rrf_k=60):
    """策略二：Reciprocal Rank Fusion。
    final = 1/(rrf_k + rank_bm25) + 1/(rrf_k + rank_emb)
    只用排名、不用分數本身，因此免正規化、免調權重，跨實作穩定。
    rrf_k=60 為文獻常用預設值。"""
    import numpy as np
    if not docs:
        return []
    bm_rank = _rank_map(_bm25_scores(query, docs))
    em_rank = _rank_map(_embedding_scores(query, docs))
    final = np.array([
        1.0 / (rrf_k + bm_rank[i]) + 1.0 / (rrf_k + em_rank[i])
        for i in range(len(docs))
    ])
    order = np.argsort(-final, kind="stable")
    return [{"id": docs[i]["id"], "score": float(final[i])} for i in order[:k]]


def hybrid_search(
    query: str,
    docs: list[dict],
    k: int = 8,
    mode: str = "rrf",
    alpha: float = 0.5,
) -> list[dict]:
    """Hybrid 檢索入口。
    docs = [{"id": str, "text": str}, ...]；回傳 [{"id", "score"}, ...]（高到低，前 k 筆）。
    mode：'weighted'（加權）或 'rrf'（排名融合）。
    """
    if mode == "weighted":
        return _weighted_search(query, docs, k, alpha=alpha)
    elif mode == "rrf":
        return _rrf_search(query, docs, k)
    else:
        raise ValueError(f"unknown hybrid mode: {mode}")
