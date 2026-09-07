from __future__ import annotations

import re
import time

from app.config import get_settings
from app.graph.state import record_step

_STATUS_PATTERNS = [
    ("married_joint", r"married[\s-]*(?:filing[\s-]*)?joint"),
    ("married_separate", r"married[\s-]*(?:filing[\s-]*)?separate"),
    ("head_of_household", r"head of household"),
    ("single", r"\bsingle\b"),
]
_MONEY = re.compile(r"\$?\s*(\d[\d,]*)(?:\.\d+)?\s*(k)?", re.I)
_DEP = re.compile(r"(\d+)\s+dependent", re.I)
_YEAR = re.compile(r"\b(20\d{2})\b")


def _extract_profile(text: str) -> dict:
    low = text.lower()
    status = next((s for s, pat in _STATUS_PATTERNS if re.search(pat, low)), "single")
    m = _MONEY.search(text)
    income = 0.0
    if m:
        income = float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1)
    dep = int(_DEP.search(text).group(1)) if _DEP.search(text) else 0
    ym = _YEAR.search(text)
    year = int(ym.group(1)) if ym else get_settings().tax_year
    return {"filing_status": status, "gross_income": income,
            "dependents": dep, "tax_year": year}


def plan(state: dict) -> dict:
    start = time.perf_counter()
    route = state.get("route", "rag_only")
    if route == "rag_only":
        return {"subtasks": ["retrieve"], "tax_profile": None,
                **record_step("plan", start, "no calc needed")}
    profile = _extract_profile(state["question"])
    subtasks = (["retrieve", "calculate", "synthesize"]
                if route == "rag_plus_calc" else ["calculate", "synthesize"])
    return {"subtasks": subtasks, "tax_profile": profile,
            **record_step("plan", start, f"profile={profile}")}
