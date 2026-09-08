from __future__ import annotations

import re

from app.llm.provider import get_llm
from app.rag.state import RagState

_SYS = "Rewrite the user's tax question into short standalone search queries, one per line."

# The IRS pubs use fixed terms of art that user phrasing rarely matches. A small
# deterministic map bridges the gap so retrieval doesn't depend on the local
# model guessing the right vocabulary.
_DOMAIN_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bbrackets?\b|\bmarginal rate\b|\btax rate", re.I),
     "2025 Tax Rate Schedules Schedule X Y Z tax computation single"),
    (re.compile(r"\bstandard deduction\b", re.I),
     "2025 Standard Deduction Tables amount by filing status"),
    (re.compile(r"\bestimated tax\b|\bquarterly\b|\bpay as you go\b", re.I),
     "estimated tax payment due dates April June September January"),
    (re.compile(r"\bhead of household\b", re.I),
     "qualifying person keeping up a home cost test head of household"),
]


def expand_query(state: RagState) -> dict:
    question = state["question"]
    rounds = state.get("rounds", 0) + 1
    queries = [question]
    try:
        raw = get_llm().complete(f"Question: {question}", system=_SYS, max_tokens=96)
        for line in raw.splitlines():
            line = line.strip("-* \t")
            if line and line.lower() != question.lower() and len(queries) < 3:
                queries.append(line)
    except Exception:  # noqa: BLE001 - retrieval must not crash on LLM failure
        pass
    for pat, hint in _DOMAIN_HINTS:
        if len(queries) >= 5:
            break
        if pat.search(question) and hint not in queries:
            queries.append(hint)
    if rounds >= 2:
        queries.append(f"{question} rule amount table IRS publication")

    from app.observability.logging import log_event

    log_event("expand_query", "expanded", question=question, queries=queries,
              n=len(queries), round=rounds)
    return {"queries": queries, "rounds": rounds}
