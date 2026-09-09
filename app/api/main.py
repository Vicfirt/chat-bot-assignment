from __future__ import annotations

import json
import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph.main_graph import run_agent, run_agent_stream
from app.observability.logging import configure_logging, log_event, new_request_id, set_request_id
from app.observability.metrics import (
    RETRIEVAL_CHUNKS,
    metrics_asgi_app,
    record_cache,
    record_context,
    record_request,
    record_steps,
)
from app.rag import cache

configure_logging()
app = FastAPI(title="Agentic RAG Tax Chatbot")


def _index_size() -> int:
    """Chunk count in the embedded Chroma index, or -1 if it can't be read."""
    try:
        from app.rag.retriever import get_retriever

        return get_retriever()._dense.count()
    except Exception:  # noqa: BLE001
        return -1


@app.on_event("startup")
def _warn_if_no_index() -> None:
    n = _index_size()
    if n <= 0:
        log_event("api", "index.empty", level=logging.WARNING, chunks=n,
                  hint="run `make ingest` (or `python -m app.ingest.build_index`) "
                       "before serving — retrieval will return no citations")


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
    retrieval_funnel: dict = Field(default_factory=dict)
    cached: bool = False


def _response_key(question: str) -> tuple:
    return (cache.normalize_question(question), cache.index_fingerprint())


def _cached_response(question: str, history: list) -> dict | None:
    """End-to-end response cache. Unsafe with conversation history (the answer
    depends on it) and disabled when caching is off."""
    if history or not cache.enabled():
        return None
    hit = cache._response_cache.get(_response_key(question))
    record_cache("response", hit is not None)
    return hit


@app.get("/health")
def health() -> dict:
    n = _index_size()
    return {
        "status": "ok" if n > 0 else "degraded",
        "llm_mode": get_settings().llm_mode,
        "index_chunks": n,
        **({} if n > 0 else {"detail": "vector index empty — run `make ingest`"}),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    start = time.perf_counter()
    set_request_id(new_request_id())
    history = [m.model_dump() for m in req.chat_history]
    log_event("api", "request.received", question=req.question, history_len=len(history))

    cached = _cached_response(req.question, history)
    if cached is not None:
        total_ms = round((time.perf_counter() - start) * 1000, 1)
        record_request(cached["route"], "ok", total_ms / 1000.0)
        log_event("api", "request.completed", route=cached["route"], total_ms=total_ms,
                  n_citations=len(cached["citations"]), cached=True)
        return ChatResponse(**{
            **cached, "cached": True,
            "timings": {**cached.get("timings", {}), "total_ms": total_ms},
        })

    try:
        state = run_agent(req.question, history)
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

    payload = {
        "answer": state.get("final_answer", ""),
        "citations": citations,
        "route": route,
        "steps": steps,
        "timings": {"total_ms": total_ms,
                    "per_node_ms": {s["node"]: s["duration_ms"] for s in steps}},
        "low_confidence": low_conf,
        "guardrail": state.get("guardrail", {}),
        "retrieval_funnel": state.get("retrieval_funnel", {}),
    }
    if not history and cache.enabled() and state.get("final_answer"):
        cache._response_cache.put(_response_key(req.question), payload)
    return ChatResponse(**payload)


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Server-sent events: one `{"type":"step", ...}` per graph node as it
    finishes, then a final `{"type":"final", ...}` with the answer."""
    start = time.perf_counter()
    set_request_id(new_request_id())
    history = [m.model_dump() for m in req.chat_history]
    log_event("api", "request.received", question=req.question, history_len=len(history),
              mode="stream")

    def gen():
        cached = _cached_response(req.question, history)
        if cached is not None:
            total_ms = round((time.perf_counter() - start) * 1000, 1)
            record_request(cached["route"], "ok", total_ms / 1000.0)
            log_event("api", "request.completed", route=cached["route"],
                      total_ms=total_ms, n_citations=len(cached["citations"]),
                      cached=True, mode="stream")
            yield _sse({"type": "step", "node": "cache", "duration_ms": total_ms,
                        "summary": "response cache hit"})
            yield _sse({"type": "final", "answer": cached["answer"],
                        "citations": cached["citations"], "route": cached["route"],
                        "low_confidence": cached["low_confidence"],
                        "guardrail": cached["guardrail"],
                        "retrieval_funnel": cached["retrieval_funnel"],
                        "cached": True, "total_ms": total_ms})
            return

        acc: dict = {}
        step_list: list[dict] = []
        try:
            for chunk in run_agent_stream(req.question, history):
                for _node, delta in chunk.items():
                    for s in delta.get("steps", []):
                        record_steps([s])
                        step_list.append(s)
                        yield _sse({"type": "step", **s})
                    acc.update({k: v for k, v in delta.items() if k != "steps"})
        except Exception as e:  # noqa: BLE001
            record_request("unknown", "error", time.perf_counter() - start)
            log_event("api", "request.failed", error=str(e), level=logging.ERROR, mode="stream")
            yield _sse({"type": "error", "message": str(e)})
            return

        total_ms = round((time.perf_counter() - start) * 1000, 1)
        route = acc.get("route", "unknown")
        citations = acc.get("citations", []) or []
        low_conf = bool(acc.get("validation", {}).get("low_confidence", False))
        record_request(route, "ok", total_ms / 1000.0)
        record_context(acc.get("rag_context", ""))
        RETRIEVAL_CHUNKS.observe(len(citations))
        log_event("api", "request.completed", route=route, total_ms=total_ms,
                  n_citations=len(citations), low_confidence=low_conf, mode="stream",
                  guardrail_violations=acc.get("guardrail", {}).get("violations", []))
        final = {
            "answer": acc.get("final_answer", ""),
            "citations": citations,
            "route": route,
            "low_confidence": low_conf,
            "guardrail": acc.get("guardrail", {}),
            "retrieval_funnel": acc.get("retrieval_funnel", {}),
        }
        if not history and cache.enabled() and final["answer"]:
            cache._response_cache.put(_response_key(req.question), {
                **final, "steps": step_list,
                "timings": {"total_ms": total_ms,
                            "per_node_ms": {s["node"]: s["duration_ms"] for s in step_list}},
            })
        yield _sse({"type": "final", **final, "total_ms": total_ms})

    return StreamingResponse(gen(), media_type="text/event-stream")


app.mount("/metrics", metrics_asgi_app())
