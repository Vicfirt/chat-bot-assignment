from __future__ import annotations

import re
import time

from app.graph.state import ROUTES, record_step
from app.llm.provider import get_llm

# A real money amount: a "$" figure, a comma-grouped number (85,000), or "60k".
_MONEY = re.compile(r"\$\s?\d[\d,]*|\b\d{1,3}(,\d{3})+\b|\b\d+k\b", re.I)
_CALC_HINT = re.compile(
    r"\b(how much|owe|estimat\w*|calculat\w*|effective|marginal|"
    r"tax (on|rate|bill|liability)|do i pay|what.s my tax|"
    r"my (federal |state |income |total )*tax(es)?)\b",
    re.I,
)
# Any hint that the question is actually about U.S. federal individual income
# tax. Small local models sometimes label a plainly unrelated question
# ("capital of France") as in-scope; if none of this vocabulary is present we
# override to out_of_scope.
_TAX_VOCAB = re.compile(
    r"\b(tax(es|able|payer|ation)?|deduct\w*|withhold\w*|filing|file[sd]?\s+(a\s+)?return|"
    r"depend[ae]nts?|exemptions?|income|irs|refunds?|brackets?|credits?|agi|1040|w-?2|"
    r"publication|estimated|itemiz\w*|head of household|married|spouse|"
    r"qualifying (child|relative|person|surviving)|wages?|earned income|capital gains?)\b",
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
    llm_route = route
    calc_signal = bool(_MONEY.search(q) and _CALC_HINT.search(q))
    if route == "rag_only" and calc_signal:
        route = "rag_plus_calc"
    # No tax vocabulary and no calculation signal -> it isn't a tax question.
    if route != "out_of_scope" and not (_TAX_VOCAB.search(q) or calc_signal):
        route = "out_of_scope"

    from app.observability.logging import log_event

    log_event("triage", "routed", question=q, llm_route=llm_route, route=route,
              overridden=llm_route != route)
    return {"route": route, **record_step("triage", start, f"route={route}")}


def route_after_triage(state: dict) -> str:
    return state["route"]
