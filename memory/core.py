"""把 store + bm25 接起來，對外暴露 capture / retrieve（自動評分與 benchmark 會呼叫）。"""
from __future__ import annotations
import hashlib
import os
import time
from pathlib import Path

from .store import JsonStore
from .bm25 import bm25_search

_store_path = os.environ.get("PI_MEMORY_PATH") or str(Path.home() / ".pi-memory.json")
_store: JsonStore | None = None


def _get_store() -> JsonStore:
    global _store
    if _store is None:
        _store = JsonStore(_store_path)
    return _store


def set_memory_path(path: str) -> None:
    """測試 / bridge 用來指定記憶檔位置（會重新載入）。"""
    global _store_path, _store
    _store_path = path
    _store = JsonStore(_store_path)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    """粗估 token 數，約 = 字元數 / 4。"""
    return max(1, len(text) // 4)


def make_observation(summary: str, session_id: str = "s", tool_name: str = "remember",
                     tags: list[str] | None = None) -> dict:
    return {
        "id": sha256(summary),
        "sessionId": session_id,
        "timestamp": int(time.time() * 1000),
        "toolName": tool_name,
        "summary": summary,
        "tags": tags or [],
    }


def capture(obs: dict) -> None:
    """(a)(b) Capture + Store。"""
    _get_store().add(obs)


def retrieve(query: str, k: int) -> list[dict]:
    """(c) Retrieve：找出與 query 最相關的前 K 筆 Observation。

    預設走確定性的純 BM25（核心測試與一般情況）。
    任務二：設定環境變數 RETRIEVAL_MODE=weighted 或 rrf 時改走 hybrid_search，
    benchmark 與 Pi demo 皆透過此開關使用同一條檢索路徑，確保一致與可重現。
    未設定時行為與原本完全相同，核心單元測試不受影響。"""
    all_obs = _get_store().all()
    docs = [{"id": o["id"], "text": " ".join([o["summary"], *o.get("tags", [])])} for o in all_obs]

    mode = os.environ.get("RETRIEVAL_MODE", "bm25")
    if mode == "bm25":
        ranked = bm25_search(query, docs, k)
    else:
        # 延遲 import：只有真的走 hybrid 才載入，未裝 sentence-transformers 不影響核心
        from .hybrid import hybrid_search
        ranked = hybrid_search(query, docs, k, mode=mode)

    by_id = {o["id"]: o for o in all_obs}
    return [by_id[r["id"]] for r in ranked if r["id"] in by_id]


def build_injection(query: str, token_budget: int = 2000, k: int = 8) -> str:
    """(d) Inject：把檢索結果在 token 預算內組成一段文字。"""
    hits = retrieve(query, k)
    header = "[記憶 - 來自過去的 session]"
    lines, used = [], estimate_tokens(header)
    for h in hits:
        line = f"- {h['summary']}"
        cost = estimate_tokens(line)
        if used + cost > token_budget:
            break
        lines.append(line)
        used += cost
    return "\n".join([header, *lines]) if lines else ""
