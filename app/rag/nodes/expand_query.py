from __future__ import annotations

from app.llm.provider import get_llm
from app.rag.state import RagState

_SYS = "Rewrite the user's tax question into short standalone search queries, one per line."


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
    if rounds >= 2:
        queries.append(f"{question} rule amount table IRS publication")
    return {"queries": queries, "rounds": rounds}
