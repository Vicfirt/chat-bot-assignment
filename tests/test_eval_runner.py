from eval.run_eval import load_eval_set, score_item


def test_load_eval_set_has_15_items():
    items = load_eval_set("eval/questions.yaml")
    assert len(items) == 15
    assert {i["route_expected"] for i in items} >= {"rag_only", "needs_calc", "rag_plus_calc", "out_of_scope"}


def test_score_item_matches_route_and_number():
    item = {"id": "q08", "route_expected": "rag_plus_calc",
            "reference_answer": "tax about $10,541", "expected_sources": [{"pub": "Pub. 501"}],
            "expected_number": 10541, "tolerance": 25}
    state = {"route": "rag_plus_calc", "citations": [{"pub": "Pub. 501"}],
             "calc_result": {"total_tax": 10541.0}, "final_answer": "You owe 10,541. [Pub. 501]"}
    s = score_item(item, state)
    assert s["route_ok"] and s["retrieval_hit"] and s["has_citation"]
    assert s["number_ok"] is True
