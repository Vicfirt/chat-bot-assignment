from __future__ import annotations

from app.config import get_settings
from app.rag.state import RagState


def grade_docs(state: RagState) -> dict:
    # Candidates arrive already fused + reranked + score-filtered by the
    # retriever; this node only enforces the floor and the min-docs count.
    s = get_settings()
    hits = state.get("raw_hits", [])
    kept = [h for h in hits if h["score"] >= s.grade_min_score]
    if len(kept) < s.min_docs:
        kept = hits[: s.min_docs]
    return {"graded_hits": kept}


def route_after_grade(state: RagState) -> str:
    min_docs = get_settings().min_docs
    if len(state.get("graded_hits", [])) < min_docs and state.get("rounds", 1) < 2:
        return "expand"
    return "assemble"
