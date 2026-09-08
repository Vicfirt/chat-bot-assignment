from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.graph.nodes.calculate import calculate
from app.graph.nodes.guardrails import guardrails
from app.graph.nodes.plan import plan
from app.graph.nodes.retrieve import retrieve
from app.graph.nodes.synthesize import synthesize
from app.graph.nodes.triage import route_after_triage, triage
from app.graph.nodes.validate import route_after_validate, validate
from app.graph.state import AgentState

_compiled = None


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("triage", triage)
    g.add_node("plan", plan)
    g.add_node("retrieve", retrieve)
    g.add_node("calculate", calculate)
    g.add_node("synthesize", synthesize)
    g.add_node("guardrails", guardrails)
    g.add_node("validate", validate)

    g.set_entry_point("triage")
    g.add_conditional_edges("triage", route_after_triage, {
        "rag_only": "retrieve",
        "needs_calc": "plan",
        "rag_plus_calc": "plan",
        "out_of_scope": "synthesize",
    })
    g.add_conditional_edges("plan", lambda s: s["route"], {
        "needs_calc": "calculate",
        "rag_plus_calc": "retrieve",
    })
    g.add_edge("retrieve", "calculate")
    g.add_edge("calculate", "synthesize")
    g.add_edge("synthesize", "guardrails")
    g.add_edge("guardrails", "validate")
    g.add_conditional_edges("validate", route_after_validate,
                            {"retry": "retrieve", "end": END})
    return g.compile()


def run_agent(question: str, chat_history: list[dict] | None = None,
              callbacks: list | None = None) -> dict:
    global _compiled
    if _compiled is None:
        _compiled = build_graph()

    from app.observability.logging import configure_logging, get_request_id, new_request_id, set_request_id

    configure_logging()
    if get_request_id() == "-":
        set_request_id(new_request_id())

    return _compiled.invoke(
        {"question": question, "chat_history": chat_history or [], "retry_count": 0},
        config={"callbacks": callbacks or [], "recursion_limit": 25},
    )
