from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.rag.nodes.assemble_context import assemble_context
from app.rag.nodes.expand_query import expand_query
from app.rag.nodes.grade_docs import grade_docs, route_after_grade
from app.rag.nodes.vector_search import vector_search
from app.rag.state import RagState

_compiled = None


def build_rag_subgraph():
    g = StateGraph(RagState)
    g.add_node("expand_query", expand_query)
    g.add_node("vector_search", vector_search)
    g.add_node("grade_docs", grade_docs)
    g.add_node("assemble_context", assemble_context)

    g.set_entry_point("expand_query")
    g.add_edge("expand_query", "vector_search")
    g.add_edge("vector_search", "grade_docs")
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
