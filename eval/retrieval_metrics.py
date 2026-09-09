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


def _rel(hit: dict, gold: set[Key], tol: int) -> bool:
    """A hit is relevant if it is on a gold page, or within `tol` pages of one
    in the same publication (content spans page boundaries)."""
    pub, page = hit_key(hit)
    return any((pub, page + d) in gold for d in range(-tol, tol + 1))


def precision_at_k(hits: list[dict], gold: set[Key], k: int, tol: int = 0) -> float | None:
    top = hits[:k]
    if not gold or not top:
        return None
    return sum(_rel(h, gold, tol) for h in top) / len(top)


def recall_at_k(hits: list[dict], gold: set[Key], k: int, tol: int = 0) -> float | None:
    if not gold:
        return None
    covered = {g for g in gold for h in hits[:k]
               if h.get("pub", "?") == g[0] and abs(int(h.get("page", 0) or 0) - g[1]) <= tol}
    return len(covered) / len(gold)


def reciprocal_rank(hits: list[dict], gold: set[Key], tol: int = 0) -> float | None:
    if not gold:
        return None
    for i, h in enumerate(hits, 1):
        if _rel(h, gold, tol):
            return 1.0 / i
    return 0.0


def _run_subgraph(question: str, tax_profile: dict | None = None):
    from app.rag.subgraph import build_rag_subgraph

    if _run_subgraph.compiled is None:
        _run_subgraph.compiled = build_rag_subgraph()
    return _run_subgraph.compiled.invoke(
        {"question": question, "chat_history": [],
         "tax_profile": tax_profile, "rounds": 0}
    )


_run_subgraph.compiled = None


def evaluate_item(item: dict, k: int) -> dict | None:
    gold = gold_keys(item)
    if not gold:
        return None
    # Mirror the real graph: `plan` only produces a tax_profile on the calc routes.
    profile = None
    if item.get("route_expected") in ("needs_calc", "rag_plus_calc"):
        from app.graph.nodes.plan import _extract_profile

        profile = _extract_profile(item["question"])
    state = _run_subgraph(item["question"], profile)
    fused = state.get("raw_hits", [])                       # pre-rerank (RRF order)
    reranked = state.get("reranked_hits", fused)            # post cross-encoder
    graded = state.get("graded_hits", reranked)             # kept for the prompt
    return {
        "id": item["id"],
        "precision_at_k": precision_at_k(reranked, gold, k),
        "recall_at_k": recall_at_k(reranked, gold, k),
        "mrr": reciprocal_rank(reranked, gold),
        "precision_at_k_tol1": precision_at_k(reranked, gold, k, tol=1),
        "recall_at_k_tol1": recall_at_k(reranked, gold, k, tol=1),
        "mrr_tol1": reciprocal_rank(reranked, gold, tol=1),
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
        "precision_at_k_tol1": _avg([r["precision_at_k_tol1"] for r in rows]),
        "recall_at_k_tol1": _avg([r["recall_at_k_tol1"] for r in rows]),
        "mrr_tol1": _avg([r["mrr_tol1"] for r in rows]),
        "context_precision": _avg([r["context_precision"] for r in rows]),
        "mrr_fused": _avg([r["rr_fused"] for r in rows]),
        "mrr_reranked": _avg([r["rr_reranked"] for r in rows]),
        "hit_rate_fused": round(mean(r["hit_fused_at_k"] for r in rows), 3),
        "hit_rate_reranked": round(mean(r["hit_reranked_at_k"] for r in rows), 3),
    }
    agg["rerank_mrr_lift"] = round(agg["mrr_reranked"] - agg["mrr_fused"], 3)
    agg["rerank_hit_lift"] = round(agg["hit_rate_reranked"] - agg["hit_rate_fused"], 3)
    return {"rows": rows, "aggregate": agg}
