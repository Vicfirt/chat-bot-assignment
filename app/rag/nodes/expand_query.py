from __future__ import annotations

import re

from app.config import get_settings
from app.llm.provider import get_llm
from app.rag.state import RagState

_SYS = "Rewrite the user's tax question into short standalone search queries, one per line."

_STATUS_PHRASE = {
    "single": "single",
    "married_joint": "married filing jointly",
    "married_separate": "married filing separately",
    "head_of_household": "head of household",
}


def _profile_queries(profile: dict | None) -> list[str]:
    """Deterministic queries built from the extracted tax profile. A calc
    question ("how much do I owe on $85k, single?") needs the standard-deduction
    table and the rate schedule, but that phrasing matches neither — so name
    them explicitly instead of hoping the LLM rewrite does."""
    if not profile:
        return []
    year = profile.get("tax_year") or get_settings().tax_year
    status = _STATUS_PHRASE.get(profile.get("filing_status", ""), "")
    return [
        f"{year} standard deduction amount {status}".strip(),
        f"{year} tax rate schedule {status} Schedule X Y Z".strip(),
    ]
_CONDENSE_SYS = (
    "Given the recent conversation and a follow-up question, rewrite the follow-up "
    "as a single standalone question that needs no prior context. Reply with the "
    "rewritten question only."
)


def _standalone_question(question: str, chat_history: list[dict]) -> str:
    """Fold the last turns into the question so a follow-up ('what about for
    married?') retrieves correctly. No-op when there is no history."""
    turns = [m for m in (chat_history or []) if m.get("content")]
    if not turns:
        return question
    convo = "\n".join(f"{m.get('role', 'user')}: {m['content']}" for m in turns[-4:])
    try:
        rewritten = get_llm().complete(
            f"Conversation:\n{convo}\n\nFollow-up: {question}",
            system=_CONDENSE_SYS, max_tokens=64,
        ).strip().splitlines()[0].strip()
    except Exception:  # noqa: BLE001 - retrieval must not crash on LLM failure
        return question
    # Reject an unusable rewrite (empty, the dummy LLM's echo, or a runaway).
    if not rewritten or rewritten.startswith("[dummy]") or len(rewritten) > 4 * len(question) + 80:
        return question
    return rewritten

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
    raw_question = state["question"]
    question = _standalone_question(raw_question, state.get("chat_history", []))
    rounds = state.get("rounds", 0) + 1

    queries = [question]
    for q in _profile_queries(state.get("tax_profile")):
        if q and q not in queries:
            queries.append(q)

    try:
        raw = get_llm().complete(f"Question: {question}", system=_SYS, max_tokens=96)
        for line in raw.splitlines():
            line = line.strip("-* \t")
            if (line and not line.startswith("[dummy]")
                    and line.lower() != question.lower()
                    and line not in queries and len(queries) < 5):
                queries.append(line)
    except Exception:  # noqa: BLE001 - retrieval must not crash on LLM failure
        pass

    for pat, hint in _DOMAIN_HINTS:
        if len(queries) >= 6:
            break
        if pat.search(question) and hint not in queries:
            queries.append(hint)
    if rounds >= 2:
        queries.append(f"{question} rule amount table IRS publication")

    from app.observability.logging import log_event

    log_event("expand_query", "expanded", question=raw_question,
              standalone=question if question != raw_question else None,
              queries=queries, n=len(queries), round=rounds)
    return {"queries": queries, "rounds": rounds}
