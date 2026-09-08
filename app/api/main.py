from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph.main_graph import run_agent
from app.observability.logging import configure_logging, log_event, new_request_id, set_request_id
from app.observability.metrics import (
    RETRIEVAL_CHUNKS,
    metrics_asgi_app,
    record_context,
    record_request,
    record_steps,
)
from app.observability.tracing import get_langfuse_callbacks

configure_logging()
app = FastAPI(title="Agentic RAG Tax Chatbot")


class Msg(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    chat_history: list[Msg] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict]
    route: str
    steps: list[dict]
    timings: dict
    low_confidence: bool
    guardrail: dict = Field(default_factory=dict)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_mode": get_settings().llm_mode}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    start = time.perf_counter()
    rid = new_request_id()
    set_request_id(rid)
    history = [m.model_dump() for m in req.chat_history]
    log_event("api", "request.received", question=req.question, history_len=len(history))
    try:
        state = run_agent(req.question, history, callbacks=get_langfuse_callbacks())
    except Exception as e:  # noqa: BLE001
        record_request("unknown", "error", time.perf_counter() - start)
        log_event("api", "request.failed", error=str(e), level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e)) from e

    total_ms = round((time.perf_counter() - start) * 1000, 1)
    steps = state.get("steps", [])
    route = state.get("route", "unknown")
    citations = state.get("citations", [])

    record_request(route, "ok", total_ms / 1000.0)
    record_steps(steps)
    record_context(state.get("rag_context", ""))
    RETRIEVAL_CHUNKS.observe(len(citations))
    low_conf = bool(state.get("validation", {}).get("low_confidence", False))
    log_event("api", "request.completed", route=route, total_ms=total_ms,
              n_citations=len(citations), low_confidence=low_conf,
              guardrail_violations=state.get("guardrail", {}).get("violations", []))

    return ChatResponse(
        answer=state.get("final_answer", ""),
        citations=citations,
        route=route,
        steps=steps,
        timings={"total_ms": total_ms,
                 "per_node_ms": {s["node"]: s["duration_ms"] for s in steps}},
        low_confidence=low_conf,
        guardrail=state.get("guardrail", {}),
    )


app.mount("/metrics", metrics_asgi_app())
