from __future__ import annotations

import time

from app.config import get_settings
from app.graph.state import record_step


def validate(state: dict) -> dict:
    start = time.perf_counter()
    route = state.get("route", "rag_only")
    answer = state.get("final_answer", "")
    reasons: list[str] = []

    if route in {"rag_only", "rag_plus_calc"} and not state.get("citations"):
        reasons.append("no citations")

    calc = state.get("calc_result") or {}
    if "total_tax" in calc:
        tt = calc["total_tax"]
        if str(int(tt)) not in answer.replace(",", "") and f"{tt:,.0f}" not in answer:
            reasons.append("calc total not reflected in answer")

    ok = not reasons
    retry_count = state.get("retry_count", 0)
    max_retries = get_settings().max_retries
    should_retry = (not ok) and retry_count < max_retries
    if should_retry:
        retry_count += 1
    low_conf = (not ok) and not should_retry

    return {
        "validation": {"ok": ok, "reasons": reasons,
                       "low_confidence": low_conf, "should_retry": should_retry},
        "retry_count": retry_count,
        **record_step("validate", start, "ok" if ok else f"retry={should_retry} reasons={reasons}"),
    }


def route_after_validate(state: dict) -> str:
    return "retry" if state.get("validation", {}).get("should_retry") else "end"
