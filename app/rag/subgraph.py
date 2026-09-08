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
        with time_subgraph_node(name):
            return fn(state)
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


def run_rag(question: str, chat_history: list[dict]) -> dict:
    global _compiled
    if _compiled is None:
        _compiled = build_rag_subgraph()
    final = _compiled.invoke(
        {"question": question, "chat_history": chat_history, "rounds": 0}
    )
    return {"rag_context": final.get("rag_context", ""),
            "citations": final.get("citations", [])}
