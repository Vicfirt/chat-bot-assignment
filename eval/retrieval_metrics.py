"""Retrieval-quality metrics for the RAG subgraph: Precision@k, Recall@k, MRR,
and the lift the `rerank` node adds over pure RRF fusion.

Relevance is judged at (publication, page) granularity against the
`relevant_pages` labels in `eval/questions.yaml` — coarse enough to survive
re-chunking, specific enough that "retrieved some Pub. 17 chunk" is not a hit.
"""
from __future__ import annotations

from statistics import mean

Key = tuple[str, int]


def hit_key(hit: dict) -> Key:
    return (hit.get("pub", "?"), int(hit.get("page", 0) or 0))


def gold_keys(item: dict) -> set[Key]:
    return {(g["pub"], int(g["page"])) for g in item.get("relevant_pages", [])}


def precision_at_k(hits: list[dict], gold: set[Key], k: int) -> float | None:
    top = hits[:k]
    if not gold or not top:
        return None
    return sum(hit_key(h) in gold for h in top) / len(top)


def recall_at_k(hits: list[dict], gold: set[Key], k: int) -> float | None:
    if not gold:
        return None
    return len({hit_key(h) for h in hits[:k]} & gold) / len(gold)


def reciprocal_rank(hits: list[dict], gold: set[Key]) -> float | None:
    if not gold:
        return None
    for i, h in enumerate(hits, 1):
        if hit_key(h) in gold:
            return 1.0 / i
    return 0.0


def _run_subgraph(question: str):
    from app.rag.subgraph import build_rag_subgraph

    if _run_subgraph.compiled is None:
        _run_subgraph.compiled = build_rag_subgraph()
    return _run_subgraph.compiled.invoke(
        {"question": question, "chat_history": [], "rounds": 0}
    )


_run_subgraph.compiled = None


def evaluate_item(item: dict, k: int) -> dict | None:
    gold = gold_keys(item)
    if not gold:
        return None
    state = _run_subgraph(item["question"])
    fused = state.get("raw_hits", [])                       # pre-rerank (RRF order)
    reranked = state.get("reranked_hits", fused)            # post cross-encoder
    graded = state.get("graded_hits", reranked)             # kept for the prompt
    return {
        "id": item["id"],
        "precision_at_k": precision_at_k(reranked, gold, k),
        "recall_at_k": recall_at_k(reranked, gold, k),
        "mrr": reciprocal_rank(reranked, gold),
        "context_precision": precision_at_k(graded, gold, len(graded) or 1),
        "rr_fused": reciprocal_rank(fused, gold),
        "rr_reranked": reciprocal_rank(reranked, gold),
        "hit_fused_at_k": (recall_at_k(fused, gold, k) or 0.0) > 0,
        "hit_reranked_at_k": (recall_at_k(reranked, gold, k) or 0.0) > 0,
    }


def _avg(vals: list[float | None]) -> float:
    xs = [v for v in vals if v is not None]
    return round(mean(xs), 3) if xs else 0.0


def evaluate_retrieval(items: list[dict], k: int) -> dict:
    rows = [r for r in (evaluate_item(it, k) for it in items) if r is not None]
    if not rows:
        return {"rows": [], "aggregate": {}}
    agg = {
        "k": k,
        "n_labeled": len(rows),
        "precision_at_k": _avg([r["precision_at_k"] for r in rows]),
        "recall_at_k": _avg([r["recall_at_k"] for r in rows]),
        "mrr": _avg([r["mrr"] for r in rows]),
        "context_precision": _avg([r["context_precision"] for r in rows]),
        "mrr_fused": _avg([r["rr_fused"] for r in rows]),
        "mrr_reranked": _avg([r["rr_reranked"] for r in rows]),
        "hit_rate_fused": round(mean(r["hit_fused_at_k"] for r in rows), 3),
        "hit_rate_reranked": round(mean(r["hit_reranked_at_k"] for r in rows), 3),
    }
    agg["rerank_mrr_lift"] = round(agg["mrr_reranked"] - agg["mrr_fused"], 3)
    agg["rerank_hit_lift"] = round(agg["hit_rate_reranked"] - agg["hit_rate_fused"], 3)
    return {"rows": rows, "aggregate": agg}
