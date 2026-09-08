"""In-process caches for the retrieval path. Two layers:

* query-embedding LRU  — skips re-encoding a query string the embedder has
  already seen (queries repeat heavily once expansion + domain hints run);
* RAG-subgraph result LRU — skips the whole subgraph (expansion LLM call +
  dense + BM25 + RRF + rerank) for a question already answered.

Every key carries `index_fingerprint()`, a hash of the retrieval config plus
the live chunk count, so a re-ingest or a knob change silently invalidates the
caches instead of serving stale context.
"""
from __future__ import annotations

import hashlib
import re
import threading
from collections import OrderedDict
from typing import Any

from app.config import get_settings

_WS = re.compile(r"\s+")


def normalize_question(q: str) -> str:
    return _WS.sub(" ", q.strip().lower())


class LRU:
    def __init__(self, maxsize: int) -> None:
        self.maxsize = max(1, maxsize)
        self._d: "OrderedDict[Any, Any]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> Any | None:
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                self.hits += 1
                return self._d[key]
            self.misses += 1
            return None

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self.hits = 0
            self.misses = 0

    def __len__(self) -> int:
        return len(self._d)


_embedding_cache = LRU(get_settings().cache_embedding_size)
_rag_cache = LRU(get_settings().cache_rag_size)


def enabled() -> bool:
    return get_settings().cache_enabled


def index_fingerprint() -> str:
    s = get_settings()
    parts: list[Any] = [
        s.tax_year, s.embedding_model, s.chroma_collection, s.retrieval_mode,
        s.search_k, s.min_docs, s.context_token_budget, s.rerank_enabled,
        s.rerank_model, s.rerank_top_n, s.dense_top_k, s.bm25_top_k, s.rrf_k,
        s.filter_tax_year, s.boost_tables_for_amount_queries, s.grade_min_score,
    ]
    try:
        from app.rag import retriever

        parts.append(retriever.get_retriever()._dense.count())
    except Exception:  # noqa: BLE001 - a missing index just means "no count component"
        parts.append("?")
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


def clear_all() -> None:
    _embedding_cache.clear()
    _rag_cache.clear()


def stats() -> dict:
    return {
        "embedding": {"size": len(_embedding_cache), "hits": _embedding_cache.hits,
                      "misses": _embedding_cache.misses},
        "rag": {"size": len(_rag_cache), "hits": _rag_cache.hits,
                "misses": _rag_cache.misses},
    }
