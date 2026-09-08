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


def _calc_ok(state: dict) -> dict | None:
    r = state.get("calc_result")
    return r if r and "error" not in r else None


def _figure_line(r: dict) -> str:
    return (
        f"Estimated {r['tax_year']} federal income tax: "
        f"${r['total_tax']:,.2f} on taxable income ${r['taxable_income']:,.2f} "
        f"({r['filing_status'].replace('_', ' ')}, standard deduction "
        f"${r['standard_deduction']:,.0f}). Marginal rate "
        f"{r['marginal_rate'] * 100:.0f}%, effective rate {r['effective_rate'] * 100:.1f}%."
    )


def _build_prompt(state: dict) -> str:
    lines = [f"Question: {state['question']}"]
    if state.get("rag_context"):
        lines.append("\nReference excerpts:\n" + state["rag_context"])
    if _calc_ok(state):
        lines.append(
            "\nComputed figures (already stated to the user in a preceding line — do "
            "NOT repeat the dollar totals or rates; only explain, in one short "
            "paragraph, how the result follows from the brackets and deduction):\n"
            + json.dumps(state["calc_result"], indent=2)
        )
    return "\n".join(lines)


def synthesize(state: dict) -> dict:
    start = time.perf_counter()
    if state.get("route") == "out_of_scope":
        return {"draft_answer": _OOS, "final_answer": _OOS,
                **record_step("synthesize", start, "out_of_scope canned")}

    body = get_llm().complete(_build_prompt(state), system=_SYS, max_tokens=400).strip()
    # For calculation answers the exact dollar figure comes from the deterministic
    # tool, not the model — small models mis-transcribe it. Lead with the tool's
    # own line and let the model's prose follow as explanation.
    calc = _calc_ok(state)
    answer = f"{_figure_line(calc)}\n\n{body}" if calc else body
    answer += _DISCLAIMER
    return {"draft_answer": answer, "final_answer": answer,
            **record_step("synthesize", start, f"{len(answer)} chars")}
