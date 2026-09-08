from __future__ import annotations

from dataclasses import asdict

from app.config import get_settings
from app.rag import retriever
from app.rag.state import RagState


def retrieve_candidates(state: RagState) -> dict:
    """Union the fused (dense + BM25 + RRF) candidate pools of every expanded
    query. Reranking happens in the next node."""
    k = get_settings().search_k
    retriever_inst = retriever.get_retriever()
    fetch = getattr(retriever_inst, "search_candidates", None) or retriever_inst.search
    best: dict[str, dict] = {}
    for q in state["queries"]:
        for chunk in fetch(q, k):
            d = asdict(chunk)
            prev = best.get(chunk.chunk_id)
            if prev is None or d["score"] > prev["score"]:
                best[chunk.chunk_id] = d
    return {"raw_hits": sorted(best.values(), key=lambda h: h["score"], reverse=True)}
