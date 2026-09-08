from __future__ import annotations

import re
import time

from app.graph.state import ROUTES, record_step
from app.llm.provider import get_llm

# A real money amount: a "$" figure, a comma-grouped number (85,000), or "60k".
_MONEY = re.compile(r"\$\s?\d[\d,]*|\b\d{1,3}(,\d{3})+\b|\b\d+k\b", re.I)
_CALC_HINT = re.compile(
    r"\b(how much|owe|estimat\w*|calculat\w*|effective|marginal|"
    r"tax (on|rate|bill|liability)|do i pay|what.s my tax)\b",
    re.I,
)

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
    # A dollar figure plus a "how much do I owe" phrasing always needs the
    # calculator; small local models routinely misfile these as rag_only.
    q = state["question"]
    if route == "rag_only" and _MONEY.search(q) and _CALC_HINT.search(q):
        route = "rag_plus_calc"
    return {"route": route, **record_step("triage", start, f"route={route}")}


def route_after_triage(state: dict) -> str:
    return state["route"]
