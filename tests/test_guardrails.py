from app.graph.nodes.guardrails import guardrails
from app.graph.nodes.validate import route_after_validate, validate


def _state(answer, **kw):
    return {"route": "rag_plus_calc", "final_answer": answer,
            "citations": [{"pub": "Pub. 17", "page": 97}], "retry_count": 0, **kw}


def test_redacts_ssn_from_answer():
    out = guardrails(_state("Your SSN 123-45-6789 puts you in the 12% bracket. [Pub. 17 p.97]"))
    assert "123-45-6789" not in out["final_answer"]
    assert "[redacted-ssn]" in out["final_answer"]
    assert out["guardrail"]["pii_redacted"] is True


def test_flags_citation_to_unretrieved_page():
    out = guardrails(_state("The deduction is $15,750. [Pub. 17 p.97] See also [Pub. 501 p.4].",
                            calc_result={"standard_deduction": 15750}))
    assert out["guardrail"]["unsupported_citations"] == ["[Pub. 501 p.4]"]
    assert out["guardrail"]["violations"]


def test_flags_amount_backed_by_neither_calc_nor_context():
    st = _state("Estimated tax: $5,071.50. You could also owe $9,999 in penalties. [Pub. 17 p.97]",
                calc_result={"total_tax": 5071.5}, rag_context="")
    out = guardrails(st)
    assert out["guardrail"]["ungrounded_amounts"] == ["$9,999"]


def test_clean_answer_has_no_violations():
    st = _state("Standard deduction $15,750; tax $5,071.50. [Pub. 17 p.97]",
                calc_result={"standard_deduction": 15750, "total_tax": 5071.5},
                rag_context="the 2025 standard deduction for single filers is $15,750")
    out = guardrails(st)
    assert out["guardrail"]["violations"] == []
    assert out["guardrail"]["pii_redacted"] is False
    assert out["steps"][0]["node"] == "guardrails"


def test_amount_grounded_by_context_passes():
    st = _state("The table lists $14,600 for 2024. [Pub. 17 p.97]",
                calc_result={}, rag_context="Single | $14,600\nMarried filing jointly | $29,200")
    assert guardrails(st)["guardrail"]["ungrounded_amounts"] == []


def test_validate_consumes_guardrail_violations_and_retries():
    st = {"route": "rag_only", "final_answer": "answer [Pub. 9 p.1]",
          "citations": [{"pub": "Pub. 17"}], "retry_count": 0,
          "guardrail": {"violations": ["unsupported citations: [Pub. 9 p.1]"]}}
    out = validate(st)
    assert out["validation"]["ok"] is False
    assert "unsupported citations: [Pub. 9 p.1]" in out["validation"]["reasons"]
    assert route_after_validate({**st, **out}) == "retry"
