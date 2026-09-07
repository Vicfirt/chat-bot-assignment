from __future__ import annotations

import time

from app.graph.state import ROUTES, record_step
from app.llm.provider import get_llm

_SYS = (
    "Classify a user's message about U.S. federal income tax into one label:\n"
    "rag_only: needs a rule/definition lookup only.\n"
    "needs_calc: a pure calculation from given numbers, no rule lookup.\n"
    "rag_plus_calc: needs both a rule lookup and a calculation.\n"
    "out_of_scope: not about U.S. federal individual income tax."
)


def triage(state: dict) -> dict:
    start = time.perf_counter()
    route = get_llm().classify(state["question"], list(ROUTES), system=_SYS)
    return {"route": route, **record_step("triage", start, f"route={route}")}


def route_after_triage(state: dict) -> str:
    return state["route"]
