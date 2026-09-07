from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph.main_graph import run_agent
from app.observability.metrics import RETRIEVAL_CHUNKS, metrics_asgi_app, record_request, record_steps
from app.observability.tracing import get_langfuse_callbacks

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


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_mode": get_settings().llm_mode}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    start = time.perf_counter()
    history = [m.model_dump() for m in req.chat_history]
    try:
        state = run_agent(req.question, history, callbacks=get_langfuse_callbacks())
    except Exception as e:  # noqa: BLE001
        record_request("unknown", "error", time.perf_counter() - start)
        raise HTTPException(status_code=500, detail=str(e)) from e

    total_ms = round((time.perf_counter() - start) * 1000, 1)
    steps = state.get("steps", [])
    route = state.get("route", "unknown")
    citations = state.get("citations", [])

    record_request(route, "ok", total_ms / 1000.0)
    record_steps(steps)
    RETRIEVAL_CHUNKS.observe(len(citations))

    return ChatResponse(
        answer=state.get("final_answer", ""),
        citations=citations,
        route=route,
        steps=steps,
        timings={"total_ms": total_ms,
                 "per_node_ms": {s["node"]: s["duration_ms"] for s in steps}},
        low_confidence=bool(state.get("validation", {}).get("low_confidence", False)),
    )


app.mount("/metrics", metrics_asgi_app())
