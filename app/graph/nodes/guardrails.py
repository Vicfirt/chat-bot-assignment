"""Deterministic output guardrails, run between synthesize and validate.

No model calls: redact obvious PII, and flag claims the answer is not entitled
to make — a citation to a page that was never retrieved, or a dollar amount
that traces to neither the calculator nor the retrieved context. `validate`
turns the findings into a retry / low-confidence signal.
"""
from __future__ import annotations

import re
import time

from app.graph.state import record_step

_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CITE = re.compile(r"\[\s*Pub\.?\s*(\d+)\s*[,;]?\s*p\.?\s*(\d+)\s*\]", re.I)
_AMOUNT = re.compile(r"\$\s?(\d[\d,]*(?:\.\d{1,2})?)")


def _redact_pii(text: str) -> tuple[str, bool]:
    out = _SSN.sub("[redacted-ssn]", text)
    return out, out != text


def _known_pages(state: dict) -> set[tuple[str, str]]:
    pages = set()
    for c in state.get("citations", []) or []:
        digits = re.sub(r"\D", "", str(c.get("pub", "")))
        if digits:
            pages.add((digits, str(c.get("page", "")).strip()))
    return pages


def _unsupported_citations(answer: str, known: set[tuple[str, str]]) -> list[str]:
    seen: list[str] = []
    for m in _CITE.finditer(answer):
        if (m.group(1), m.group(2)) not in known and m.group(0) not in seen:
            seen.append(m.group(0))
    return seen


def _norm_amount(raw: str) -> str:
    n = raw.replace(",", "")
    if "." in n:
        n = n.rstrip("0").rstrip(".")
    return n


def _iter_numbers(obj) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [n for v in obj.values() for n in _iter_numbers(v)]
    if isinstance(obj, (list, tuple)):
        return [n for v in obj for n in _iter_numbers(v)]
    return []


def _grounded_amounts(state: dict) -> set[str]:
    allowed: set[str] = set()
    for v in _iter_numbers(state.get("calc_result") or {}):
        allowed.add(_norm_amount(f"{v:.2f}"))
    texts = [state.get("rag_context", "") or ""]
    texts += [str(c.get("quote", "")) for c in state.get("citations", []) or []]
    for text in texts:
        for m in _AMOUNT.finditer(text):
            allowed.add(_norm_amount(m.group(1)))
    return allowed


def _ungrounded_amounts(answer: str, state: dict) -> list[str]:
    allowed = _grounded_amounts(state)
    bad: list[str] = []
    for m in _AMOUNT.finditer(answer):
        norm = _norm_amount(m.group(1))
        # Ignore small integers (counts, "$0"); only police real money claims.
        if float(norm) < 100:
            continue
        if norm not in allowed and m.group(0) not in bad:
            bad.append(m.group(0))
    return bad


def guardrails(state: dict) -> dict:
    start = time.perf_counter()
    answer = state.get("final_answer", "")
    answer, pii_redacted = _redact_pii(answer)

    violations: list[str] = []
    bad_cites = _unsupported_citations(answer, _known_pages(state))
    if bad_cites:
        violations.append(f"unsupported citations: {', '.join(bad_cites)}")
    bad_amounts = _ungrounded_amounts(answer, state)
    # On a pure rule-lookup answer an unlisted dollar figure is usually the model
    # restating a published threshold ($1,000 estimated-tax floor, additional
    # standard-deduction amounts) that our truncated excerpt just didn't include
    # — record it, but don't force a retry. On calc routes it stays a violation.
    if bad_amounts and state.get("route") != "rag_only":
        violations.append(f"ungrounded amounts: {', '.join(bad_amounts)}")

    guardrail = {
        "pii_redacted": pii_redacted,
        "unsupported_citations": bad_cites,
        "ungrounded_amounts": bad_amounts,
        "violations": violations,
    }
    summary = "clean" if not violations and not pii_redacted else (
        f"pii_redacted={pii_redacted} violations={violations}")
    return {"final_answer": answer, "guardrail": guardrail,
            **record_step("guardrails", start, summary)}
