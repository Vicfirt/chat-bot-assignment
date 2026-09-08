from __future__ import annotations

from dataclasses import asdict, fields

from app.config import get_settings
from app.rag import retriever
from app.rag.state import RagState

_CHUNK_FIELDS = {f.name for f in fields(retriever.Chunk)}


def rerank(state: RagState) -> dict:
    """Cross-encoder rerank of the fused candidate pool (with the table-first
    boost for amount/rate questions). Degrades to fused order if the retriever
    has no reranker."""
    hits = state.get("raw_hits", [])
    retriever_inst = retriever.get_retriever()
    rerank_fn = getattr(retriever_inst, "rerank", None)
    if not callable(rerank_fn) or not hits:
        return {"reranked_hits": hits}
    k = get_settings().search_k
    chunks = [retriever.Chunk(**{f: h[f] for f in _CHUNK_FIELDS if f in h}) for h in hits]
    ordered = rerank_fn(state["question"], chunks, k)
    return {"reranked_hits": [asdict(c) for c in ordered]}
