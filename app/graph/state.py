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
    subtasks: list[str]
    tax_profile: dict | None
    rag_context: str
    citations: list[dict]
    calc_result: dict | None
    draft_answer: str
    final_answer: str
    guardrail: dict
    validation: dict
    retry_count: int
    steps: Annotated[list[dict], _append]


def record_step(node: str, start: float, summary: str) -> dict:
    return {"steps": [{"node": node,
                       "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                       "summary": summary}]}
