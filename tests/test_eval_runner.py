from eval.run_eval import load_eval_set, score_item


def test_load_eval_set_covers_end_to_end_routes():
    items = load_eval_set("eval/questions.yaml")
    assert 10 <= len(items) <= 20
    # The set exercises the three routes that run end to end. `needs_calc` (pure
    # calculation, no retrieval) is validated at the node level in
    # tests/test_graph_triage.py instead — with a real 3B the label is unstable
    # on calc questions and the triage guard deliberately biases dollar-amount
    # questions to rag_plus_calc so every returned figure carries a citation.
    assert {i["route_expected"] for i in items} >= {"rag_only", "rag_plus_calc", "out_of_scope"}


def test_score_item_matches_route_and_number():
    item = {"id": "q08", "route_expected": "rag_plus_calc",
            "reference_answer": "tax about $10,541", "expected_sources": [{"pub": "Pub. 501"}],
            "expected_number": 10314, "tolerance": 25}
    state = {"route": "rag_plus_calc", "citations": [{"pub": "Pub. 501"}],
             "calc_result": {"total_tax": 10314.0}, "final_answer": "You owe 10,314. [Pub. 501]"}
    s = score_item(item, state)
    assert s["route_ok"] and s["retrieval_hit"] and s["has_citation"]
    assert s["number_ok"] is True
