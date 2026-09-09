from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.observability.metrics import time_subgraph_node
from app.rag.nodes.assemble_context import assemble_context
from app.rag.nodes.expand_query import expand_query
from app.rag.nodes.grade_docs import grade_docs, route_after_grade
from app.rag.nodes.rerank import rerank
from app.rag.nodes.retrieve_candidates import retrieve_candidates
from app.rag.state import RagState

_compiled = None


def _timed(name, fn):
    def run(state):
        import time

        from app.observability.logging import log_event

        start = time.perf_counter()
        with time_subgraph_node(name):
            out = fn(state)
        log_event(name, "done",
                  duration_ms=round((time.perf_counter() - start) * 1000, 1))
        return out
    return run


def build_rag_subgraph():
    g = StateGraph(RagState)
    g.add_node("expand_query", _timed("expand_query", expand_query))
    g.add_node("retrieve_candidates", _timed("retrieve_candidates", retrieve_candidates))
    g.add_node("rerank", _timed("rerank", rerank))
    g.add_node("grade_docs", _timed("grade_docs", grade_docs))
    g.add_node("assemble_context", _timed("assemble_context", assemble_context))

    g.set_entry_point("expand_query")
    g.add_edge("expand_query", "retrieve_candidates")
    g.add_edge("retrieve_candidates", "rerank")
    g.add_edge("rerank", "grade_docs")
    g.add_conditional_edges(
        "grade_docs", route_after_grade,
        {"expand": "expand_query", "assemble": "assemble_context"},
    )
    g.add_edge("assemble_context", END)
    return g.compile()


def run_rag(question: str, chat_history: list[dict],
            tax_profile: dict | None = None) -> dict:
    global _compiled
    from app.observability.logging import log_event
    from app.observability.metrics import record_cache
    from app.rag import cache

    # The cache keys on the normalised question only, so it is unsafe once a
    # follow-up is resolved against conversation history (expand_query does
    # that) — bypass it whenever there is history. `tax_profile` is a pure
    # function of the question, so it needs no key of its own.
    cacheable = cache.enabled() and not chat_history
    key = (cache.normalize_question(question), cache.index_fingerprint())
    if cacheable:
        hit = cache._rag_cache.get(key)
        record_cache("rag", hit is not None)
        log_event("rag", "cache", hit=hit is not None)
        if hit is not None:
            return {"rag_context": hit["rag_context"],
                    "citations": list(hit["citations"]),
                    "funnel": {**hit.get("funnel", {}), "cached": True}}

    if _compiled is None:
        _compiled = build_rag_subgraph()
    final = _compiled.invoke(
        {"question": question, "chat_history": chat_history,
         "tax_profile": tax_profile, "rounds": 0}
    )
    out = {
        "rag_context": final.get("rag_context", ""),
        "citations": final.get("citations", []),
        "funnel": {
            "candidates": len(final.get("raw_hits", [])),
            "reranked": len(final.get("reranked_hits", [])),
            "kept": len(final.get("graded_hits", [])),
        },
    }
    log_event("rag", "context", n_citations=len(out["citations"]),
              context_words=len(out["rag_context"].split()),
              funnel=out["funnel"],
              pages=[(c.get("pub"), c.get("page")) for c in out["citations"]])
    if cacheable:
        cache._rag_cache.put(key, out)
    return out
