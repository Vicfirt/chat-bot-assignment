from __future__ import annotations

import time

from app.graph.state import record_step
from app.tools.retriever_tool import retriever_tool


def retrieve(state: dict) -> dict:
    start = time.perf_counter()
    result = retriever_tool(state["question"], state.get("chat_history", []))
    n = len(result.get("citations", []))
    return {"rag_context": result.get("rag_context", ""),
            "citations": result.get("citations", []),
            **record_step("retrieve", start, f"{n} citations")}
