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
    from app.observability.logging import log_event

    hits = state.get("raw_hits", [])
    retriever_inst = retriever.get_retriever()
    rerank_fn = getattr(retriever_inst, "rerank", None)
    if not callable(rerank_fn) or not hits:
        log_event("rerank", "skipped", used_reranker=False, n=len(hits))
        return {"reranked_hits": hits}
    k = get_settings().search_k
    chunks = [retriever.Chunk(**{f: h[f] for f in _CHUNK_FIELDS if f in h}) for h in hits]
    ordered = [asdict(c) for c in rerank_fn(state["question"], chunks, k)]
    log_event("rerank", "reranked", used_reranker=True,
              before=[(h["chunk_id"], round(h["score"], 4)) for h in hits[:5]],
              after=[(h["chunk_id"], round(h["score"], 4)) for h in ordered[:5]])
    return {"reranked_hits": ordered}
