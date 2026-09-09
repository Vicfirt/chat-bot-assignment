from __future__ import annotations

import time

from app.graph.state import record_step
from app.tools.retriever_tool import retriever_tool


def retrieve(state: dict) -> dict:
    start = time.perf_counter()
    result = retriever_tool(state["question"], state.get("chat_history", []))
    citations = result.get("citations", [])
    funnel = result.get("funnel", {})
    n = len(citations)
    summary = f"{n} citations"
    if funnel:
        summary += (" (cached)" if funnel.get("cached")
                    else f" ({funnel.get('candidates')}→{funnel.get('reranked')}"
                         f"→{funnel.get('kept')})")
    return {"rag_context": result.get("rag_context", ""),
            "citations": citations,
            "retrieval_funnel": funnel,
            **record_step("retrieve", start, summary)}
