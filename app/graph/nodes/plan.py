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
# Income amounts, most reliable first: "$85,000" / "85,000" / "$85k" / "85k".
# A bare integer is a last resort and years are excluded so "my 2025 tax" and
# "2 dependents" are never read as income.
_MONEY_STRONG = re.compile(r"\$\s*(\d[\d,]*)(?:\.\d+)?\s*(k)?|\b(\d{1,3}(?:,\d{3})+)\b|\b(\d+)\s*k\b", re.I)
_MONEY_BARE = re.compile(r"\b(\d{4,})\b")
_DEP = re.compile(r"(\d+)\s+dependent", re.I)
_YEAR = re.compile(r"\b(20\d{2})\b")


def _extract_income(text: str) -> float:
    for m in _MONEY_STRONG.finditer(text):
        digits, k1, grouped, k2 = m.groups()
        if grouped:
            return float(grouped.replace(",", ""))
        if k2:
            return float(k2) * 1000
        if digits:
            return float(digits.replace(",", "")) * (1000 if k1 else 1)
    best = 0.0
    for m in _MONEY_BARE.finditer(text):
        val = float(m.group(1))
        if 2000 <= val <= 2100:      # a year, not an amount
            continue
        best = max(best, val)
    return best


def _extract_profile(text: str) -> dict:
    low = text.lower()
    status = next((s for s, pat in _STATUS_PATTERNS if re.search(pat, low)), "single")
    income = _extract_income(text)
    dep = int(_DEP.search(text).group(1)) if _DEP.search(text) else 0
    ym = _YEAR.search(text)
    year = int(ym.group(1)) if ym else get_settings().tax_year
    return {"filing_status": status, "gross_income": income,
            "dependents": dep, "tax_year": year}


def plan(state: dict) -> dict:
    start = time.perf_counter()
    route = state.get("route", "rag_only")
    if route == "rag_only":
        return {"tax_profile": None,
                **record_step("plan", start, "no calc needed")}
    profile = _extract_profile(state["question"])
    return {"tax_profile": profile,
            **record_step("plan", start, f"profile={profile}")}
