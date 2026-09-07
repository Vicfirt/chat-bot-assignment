from __future__ import annotations

from app.config import get_settings
from app.rag.state import RagState

_THRESHOLD = 0.25


def grade_docs(state: RagState) -> dict:
    hits = state.get("raw_hits", [])
    kept = [h for h in hits if h["score"] >= _THRESHOLD]
    min_docs = get_settings().min_docs
    if len(kept) < min_docs:
        kept = hits[:min_docs]
    return {"graded_hits": kept}


def route_after_grade(state: RagState) -> str:
    min_docs = get_settings().min_docs
    if len(state.get("graded_hits", [])) < min_docs and state.get("rounds", 1) < 2:
        return "expand"
    return "assemble"
