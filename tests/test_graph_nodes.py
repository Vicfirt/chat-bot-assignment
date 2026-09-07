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
