from __future__ import annotations

import time

from app.graph.state import record_step
from app.tools.tax_calculator import estimate_tax

_KEYS = ("filing_status", "gross_income", "tax_year", "dependents")


def calculate(state: dict) -> dict:
    start = time.perf_counter()
    profile = state.get("tax_profile")
    if not profile:
        return {"calc_result": None, **record_step("calculate", start, "skipped")}
    try:
        result = estimate_tax(**{k: profile[k] for k in _KEYS})
        summary = f"total_tax={result['total_tax']}"
    except ValueError as e:
        result = {"error": str(e)}
        summary = f"error: {e}"
    return {"calc_result": result, **record_step("calculate", start, summary)}
