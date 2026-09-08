"""Deterministic U.S. federal individual income tax estimator.

Constants: IRS Rev. Proc. 2023-34 (tax year 2024) and Rev. Proc. 2024-40
(tax year 2025). This tool performs no retrieval.
"""
from __future__ import annotations

FILING_STATUSES = {"single", "married_joint", "married_separate", "head_of_household"}

STANDARD_DEDUCTION = {
    2024: {
        "single": 14_600,
        "married_joint": 29_200,
        "married_separate": 14_600,
        "head_of_household": 21_900,
    },
    2025: {
        "single": 15_000,
        "married_joint": 30_000,
        "married_separate": 15_000,
        "head_of_household": 22_500,
    },
}

# Each list: (lower_bound_of_bracket, marginal_rate), ascending.
BRACKETS = {
    2024: {
        "single": [
            (0, 0.10), (11_600, 0.12), (47_150, 0.22), (100_525, 0.24),
            (191_950, 0.32), (243_725, 0.35), (609_350, 0.37),
        ],
        "married_joint": [
            (0, 0.10), (23_200, 0.12), (94_300, 0.22), (201_050, 0.24),
            (383_900, 0.32), (487_450, 0.35), (731_200, 0.37),
        ],
        "married_separate": [
            (0, 0.10), (11_600, 0.12), (47_150, 0.22), (100_525, 0.24),
            (191_950, 0.32), (243_725, 0.35), (365_600, 0.37),
        ],
        "head_of_household": [
            (0, 0.10), (16_550, 0.12), (63_100, 0.22), (100_500, 0.24),
            (191_950, 0.32), (243_700, 0.35), (609_350, 0.37),
        ],
    },
    2025: {
        "single": [
            (0, 0.10), (11_925, 0.12), (48_475, 0.22), (103_350, 0.24),
            (197_300, 0.32), (250_525, 0.35), (626_350, 0.37),
        ],
        "married_joint": [
            (0, 0.10), (23_850, 0.12), (96_950, 0.22), (206_700, 0.24),
            (394_600, 0.32), (501_050, 0.35), (751_600, 0.37),
        ],
        "married_separate": [
            (0, 0.10), (11_925, 0.12), (48_475, 0.22), (103_350, 0.24),
            (197_300, 0.32), (250_525, 0.35), (375_800, 0.37),
        ],
        "head_of_household": [
            (0, 0.10), (17_000, 0.12), (64_850, 0.22), (103_350, 0.24),
            (197_300, 0.32), (250_500, 0.35), (626_350, 0.37),
        ],
    },
}


def estimate_tax(
    *,
    filing_status: str,
    gross_income: float,
    tax_year: int = 2025,
    dependents: int = 0,
) -> dict:
    if filing_status not in FILING_STATUSES:
        raise ValueError(f"unknown filing_status: {filing_status!r}")
    if tax_year not in BRACKETS:
        raise ValueError(f"unsupported tax_year: {tax_year}")
    if gross_income < 0 or dependents < 0:
        raise ValueError("gross_income and dependents must be non-negative")

    std = STANDARD_DEDUCTION[tax_year][filing_status]
    taxable = max(0.0, float(gross_income) - std)

    brackets = BRACKETS[tax_year][filing_status]
    breakdown: list[dict] = []
    total = 0.0
    marginal = brackets[0][1]
    for i, (lower, rate) in enumerate(brackets):
        upper = brackets[i + 1][0] if i + 1 < len(brackets) else float("inf")
        if taxable <= lower:
            break
        taxed_here = min(taxable, upper) - lower
        tax_here = taxed_here * rate
        total += tax_here
        marginal = rate
        breakdown.append(
            {"lower": float(lower), "rate": rate,
             "taxed_in_bracket": round(taxed_here, 2), "tax": round(tax_here, 2)}
        )

    total = round(total, 2)
    effective = round(total / gross_income, 4) if gross_income > 0 else 0.0
    return {
        "tax_year": tax_year,
        "filing_status": filing_status,
        "gross_income": float(gross_income),
        "standard_deduction": std,
        "taxable_income": round(taxable, 2),
        "bracket_breakdown": breakdown,
        "total_tax": total,
        "marginal_rate": marginal,
        "effective_rate": effective,
    }
