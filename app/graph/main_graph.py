from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.config import get_settings
from app.graph.nodes.calculate import calculate
from app.graph.nodes.guardrails import guardrails
from app.graph.nodes.plan import plan, route_after_plan
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
    # `rag_plus_calc` fans out into two independent subtasks; `needs_calc` runs
    # only `calculate`. Both branches rejoin at `synthesize`.
    g.add_conditional_edges("plan", route_after_plan, ["retrieve", "calculate"])
    g.add_edge("retrieve", "synthesize")
    g.add_edge("calculate", "synthesize")
    g.add_edge("synthesize", "guardrails")
    g.add_edge("guardrails", "validate")
    g.add_conditional_edges("validate", route_after_validate,
                            {"retry": "retrieve", "end": END})
    return g.compile()


def _prepare(chat_history: list[dict] | None):
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    from app.observability.logging import configure_logging, get_request_id, new_request_id, set_request_id

    configure_logging()
    if get_request_id() == "-":
        set_request_id(new_request_id())
    return {"question": "", "chat_history": chat_history or [], "retry_count": 0}


def run_agent(question: str, chat_history: list[dict] | None = None,
              callbacks: list | None = None) -> dict:
    state = {**_prepare(chat_history), "question": question}
    return _compiled.invoke(
        state, config={"callbacks": callbacks or [],
                       "recursion_limit": get_settings().graph_recursion_limit},
    )


def run_agent_stream(question: str, chat_history: list[dict] | None = None,
                     callbacks: list | None = None):
    """Yield one `{node: node_output}` dict per graph step as it completes."""
    state = {**_prepare(chat_history), "question": question}
    return _compiled.stream(
        state, config={"callbacks": callbacks or [],
                       "recursion_limit": get_settings().graph_recursion_limit},
        stream_mode="updates",
    )
