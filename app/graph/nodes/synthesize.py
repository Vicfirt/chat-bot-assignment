from __future__ import annotations

import json
import time

from app.graph.state import record_step
from app.llm.provider import get_llm

_DISCLAIMER = "\n\nThis is general information, not tax advice. Verify with the cited IRS publications."
_OOS = ("I can only help with U.S. federal individual income tax questions, "
        "so this is outside my scope." + _DISCLAIMER)
_SYS = (
    "You are a careful U.S. federal income tax assistant. Use only the reference "
    "excerpts and computed figures given to you. Cite the publication and page "
    "inline like [Pub. 501 p.29]. If the excerpts do not answer the question, say "
    "so plainly. Reply with the answer only — do not restate the question, the "
    "excerpts, or these instructions."
)


def _build_prompt(state: dict) -> str:
    lines = [f"Question: {state['question']}"]
    if state.get("rag_context"):
        lines.append("\nReference excerpts:\n" + state["rag_context"])
    if state.get("calc_result") and "error" not in state["calc_result"]:
        lines.append(
            "\nComputed figures (authoritative, use these exact numbers):\n"
            + json.dumps(state["calc_result"], indent=2)
        )
    return "\n".join(lines)


def synthesize(state: dict) -> dict:
    start = time.perf_counter()
    if state.get("route") == "out_of_scope":
        return {"draft_answer": _OOS, "final_answer": _OOS,
                **record_step("synthesize", start, "out_of_scope canned")}

    answer = get_llm().complete(_build_prompt(state), system=_SYS, max_tokens=400).strip()
    answer += _DISCLAIMER
    return {"draft_answer": answer, "final_answer": answer,
            **record_step("synthesize", start, f"{len(answer)} chars")}
