from __future__ import annotations

import json
import time

from app.graph.state import record_step
from app.llm.provider import get_llm

_DISCLAIMER = "\n\nThis is general information, not tax advice. Verify with the cited IRS publications."
_OOS = ("I can only help with U.S. federal individual income tax questions, "
        "so this is outside my scope." + _DISCLAIMER)
_SYS = ("You are a careful U.S. federal income tax assistant. Answer only from the CONTEXT "
        "and CALC blocks. Cite publications inline like [Pub. 501 p.29]. If context is "
        "insufficient, say so.")


def synthesize(state: dict) -> dict:
    start = time.perf_counter()
    if state.get("route") == "out_of_scope":
        return {"draft_answer": _OOS, "final_answer": _OOS,
                **record_step("synthesize", start, "out_of_scope canned")}

    parts = [f"QUESTION: {state['question']}"]
    if state.get("rag_context"):
        parts.append(f"CONTEXT:\n{state['rag_context']}")
    if state.get("calc_result") and "error" not in state["calc_result"]:
        parts.append(f"CALC:\n{json.dumps(state['calc_result'], indent=2)}")
    answer = get_llm().complete("\n\n".join(parts), system=_SYS, max_tokens=220).strip()
    answer += _DISCLAIMER
    return {"draft_answer": answer, "final_answer": answer,
            **record_step("synthesize", start, f"{len(answer)} chars")}
