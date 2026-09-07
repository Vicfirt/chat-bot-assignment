from __future__ import annotations

from dataclasses import asdict

from app.config import get_settings
from app.rag import retriever
from app.rag.state import RagState


def vector_search(state: RagState) -> dict:
    k = get_settings().search_k
    retriever_inst = retriever.get_retriever()
    best: dict[str, dict] = {}
    for q in state["queries"]:
        for chunk in retriever_inst.search(q, k):
            d = asdict(chunk)
            prev = best.get(chunk.chunk_id)
            if prev is None or d["score"] > prev["score"]:
                best[chunk.chunk_id] = d
    return {"raw_hits": sorted(best.values(), key=lambda h: h["score"], reverse=True)}
