from app.graph.nodes.plan import plan
from app.graph.nodes.triage import route_after_triage, triage


def test_triage_sets_route_and_step():
    out = triage({"question": "What is the standard deduction for single filers?",
                  "chat_history": []})
    assert out["route"] in {"rag_only", "rag_plus_calc"}
    assert out["steps"][0]["node"] == "triage"
    assert route_after_triage({**out}) == out["route"]


def test_triage_forces_calc_for_dollar_amount_questions(monkeypatch):
    from app.graph.nodes import triage as tri

    class _Stub:
        def classify(self, *a, **k):
            return "rag_only"

    monkeypatch.setattr(tri, "get_llm", lambda: _Stub())
    out = tri.triage({"question": "How much federal tax do I owe on $85,000, single, 2025?",
                      "chat_history": []})
    assert out["route"] == "rag_plus_calc"


def test_triage_forces_out_of_scope_when_no_tax_vocabulary(monkeypatch):
    from app.graph.nodes import triage as tri

    class _Stub:
        def classify(self, *a, **k):
            return "rag_only"

    monkeypatch.setattr(tri, "get_llm", lambda: _Stub())
    out = tri.triage({"question": "What is the capital of France?", "chat_history": []})
    assert out["route"] == "out_of_scope"


def test_plan_extracts_tax_profile():
    out = plan({"question": "How much federal tax do I owe on $85,000, filing single, 0 dependents?",
                "chat_history": [], "route": "rag_plus_calc"})
    tp = out["tax_profile"]
    assert tp["gross_income"] == 85000
    assert tp["filing_status"] == "single"
    assert tp["dependents"] == 0


def test_plan_income_ignores_year_and_dependent_count():
    out = plan({"question": "Estimate my 2025 federal tax: married filing jointly, "
                            "$150,000 income, 2 dependents.",
                "chat_history": [], "route": "rag_plus_calc"})
    tp = out["tax_profile"]
    assert tp["gross_income"] == 150000
    assert tp["tax_year"] == 2025
    assert tp["filing_status"] == "married_joint"
    assert tp["dependents"] == 2


def test_plan_noop_for_rag_only():
    out = plan({"question": "who can claim head of household?", "chat_history": [],
                "route": "rag_only"})
    assert out["tax_profile"] is None
