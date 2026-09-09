from __future__ import annotations

from typing import TypedDict


class RagState(TypedDict, total=False):
    question: str
    chat_history: list[dict]
    tax_profile: dict | None
    queries: list[str]
    raw_hits: list[dict]
    reranked_hits: list[dict]
    graded_hits: list[dict]
    rounds: int
    rag_context: str
    citations: list[dict]
