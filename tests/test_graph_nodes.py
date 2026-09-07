from app.graph.nodes.calculate import calculate
from app.graph.nodes.retrieve import retrieve


def test_calculate_uses_profile():
    out = calculate({"tax_profile": {"filing_status": "single", "gross_income": 85000,
                                     "tax_year": 2024, "dependents": 0}})
    assert out["calc_result"]["total_tax"] > 0
    assert out["steps"][0]["node"] == "calculate"


def test_calculate_noop_without_profile():
    out = calculate({"tax_profile": None})
    assert out["calc_result"] is None


def test_calculate_captures_value_error():
    out = calculate({"tax_profile": {"filing_status": "bad", "gross_income": 1,
                                     "tax_year": 2024, "dependents": 0}})
    assert "error" in out["calc_result"]


def test_retrieve_populates_context(monkeypatch):
    import app.graph.nodes.retrieve as rn

    monkeypatch.setattr(rn, "retriever_tool",
                        lambda q, h: {"rag_context": "CTX", "citations": [{"pub": "Pub. 17"}]})
    out = retrieve({"question": "q", "chat_history": []})
    assert out["rag_context"] == "CTX"
    assert out["citations"][0]["pub"] == "Pub. 17"
    assert out["steps"][0]["node"] == "retrieve"


def test_validate_flags_missing_citations():
    from app.graph.nodes.validate import route_after_validate, validate

    out = validate({"route": "rag_only", "final_answer": "answer", "citations": [],
                    "retry_count": 1})
    assert out["validation"]["ok"] is False
    assert out["validation"]["low_confidence"] is True
    assert route_after_validate({**out}) == "end"


def test_validate_retries_once_when_under_budget():
    from app.graph.nodes.validate import route_after_validate, validate

    out = validate({"route": "rag_only", "final_answer": "answer", "citations": [],
                    "retry_count": 0})
    assert out["validation"]["should_retry"] is True
    assert out["retry_count"] == 1
    assert route_after_validate({**out}) == "retry"


def test_validate_passes_with_citations_and_calc_number():
    from app.graph.nodes.validate import validate

    out = validate({"route": "rag_plus_calc",
                    "final_answer": "You owe about 10,541 dollars. [Pub. 501 p.29]",
                    "citations": [{"pub": "Pub. 501"}],
                    "calc_result": {"total_tax": 10541.0}, "retry_count": 0})
    assert out["validation"]["ok"] is True


def test_synthesize_out_of_scope_is_canned():
    from app.graph.nodes.synthesize import synthesize

    out = synthesize({"route": "out_of_scope", "question": "weather?"})
    assert "scope" in out["final_answer"].lower()
