from eval.retrieval_metrics import (
    evaluate_retrieval,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def _h(pub, page):
    return {"pub": pub, "page": page}


GOLD = {("Pub. 17", 97), ("Pub. 17", 127)}


def test_precision_and_recall_at_k():
    hits = [_h("Pub. 17", 97), _h("Pub. 501", 3), _h("Pub. 17", 127), _h("Pub. 17", 5)]
    assert precision_at_k(hits, GOLD, 4) == 0.5
    assert recall_at_k(hits, GOLD, 4) == 1.0
    assert recall_at_k(hits, GOLD, 1) == 0.5


def test_page_tolerance_counts_adjacent_pages():
    hits = [_h("Pub. 17", 96), _h("Pub. 17", 128)]   # neighbours of gold 97 / 127
    assert recall_at_k(hits, GOLD, 2, tol=0) == 0.0
    assert recall_at_k(hits, GOLD, 2, tol=1) == 1.0
    assert reciprocal_rank(hits, GOLD, tol=1) == 1.0
    assert precision_at_k(hits, GOLD, 2, tol=1) == 1.0
    # different pub is never a tolerant hit
    assert reciprocal_rank([_h("Pub. 501", 97)], GOLD, tol=1) == 0.0


def test_mrr_uses_first_relevant_rank():
    hits = [_h("Pub. 501", 3), _h("Pub. 17", 97)]
    assert reciprocal_rank(hits, GOLD) == 0.5
    assert reciprocal_rank([_h("Pub. 505", 1)], GOLD) == 0.0


def test_unlabeled_item_returns_none():
    assert precision_at_k([_h("Pub. 17", 97)], set(), 3) is None
    assert recall_at_k([], GOLD, 3) == 0.0


def test_evaluate_retrieval_reports_rerank_lift(monkeypatch):
    import eval.retrieval_metrics as m

    # Fused order buries the gold page at rank 3; rerank lifts it to rank 1.
    states = {
        "q": {
            "raw_hits": [_h("Pub. 501", 3), _h("Pub. 505", 1), _h("Pub. 17", 97)],
            "reranked_hits": [_h("Pub. 17", 97), _h("Pub. 501", 3)],
            "graded_hits": [_h("Pub. 17", 97), _h("Pub. 501", 3)],
        }
    }
    monkeypatch.setattr(m, "_run_subgraph", lambda q, profile=None: states["q"])
    items = [{"id": "t1", "question": "q", "relevant_pages": [{"pub": "Pub. 17", "page": 97}]}]
    out = evaluate_retrieval(items, k=5)
    agg = out["aggregate"]
    assert agg["mrr_fused"] == round(1 / 3, 3)
    assert agg["mrr_reranked"] == 1.0
    assert agg["rerank_mrr_lift"] == round(1 - 1 / 3, 3)
    assert agg["n_labeled"] == 1
