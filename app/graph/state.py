from __future__ import annotations

import time
from typing import Annotated, TypedDict

ROUTES = ("rag_only", "needs_calc", "rag_plus_calc", "out_of_scope")


def _append(a: list | None, b: list | None) -> list:
    return (a or []) + (b or [])


class AgentState(TypedDict, total=False):
    question: str
    chat_history: list[dict]
    route: str
    tax_profile: dict | None
    rag_context: str
    citations: list[dict]
    retrieval_funnel: dict
    calc_result: dict | None
    draft_answer: str
    final_answer: str
    guardrail: dict
    validation: dict
    retry_count: int
    steps: Annotated[list[dict], _append]


def record_step(node: str, start: float, summary: str) -> dict:
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    from app.observability.logging import log_event

    log_event(node, "done", duration_ms=duration_ms, summary=summary)
    return {"steps": [{"node": node, "duration_ms": duration_ms, "summary": summary}]}
