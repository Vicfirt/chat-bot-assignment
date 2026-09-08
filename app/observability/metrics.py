from __future__ import annotations

import time
from contextlib import contextmanager

from prometheus_client import Counter, Histogram, make_asgi_app

REQUEST_DURATION = Histogram("rag_request_duration_seconds",
                             "End-to-end /chat duration", ["route"])
NODE_DURATION = Histogram("rag_node_duration_seconds", "Main-graph per-node duration", ["node"])
REQUESTS_TOTAL = Counter("rag_requests_total", "Chat requests", ["route", "status"])
VALIDATE_RETRIES = Counter("rag_validate_retries_total", "Validate-triggered retries")
RETRIEVAL_CHUNKS = Histogram("rag_retrieval_chunks", "Citations returned per request")

# RAG subgraph internals (not carried in the main graph's `steps`).
SUBGRAPH_NODE_DURATION = Histogram(
    "rag_subgraph_node_duration_seconds", "RAG subgraph per-node duration", ["node"])
CONTEXT_WORDS = Histogram(
    "rag_context_words", "Assembled RAG context size in words",
    buckets=(50, 100, 200, 400, 600, 900, 1500, 3000))

# LLM provider.
LLM_CALLS = Counter("rag_llm_calls_total", "LLM calls", ["op", "provider"])
LLM_DURATION = Histogram("rag_llm_duration_seconds", "LLM call wall time", ["op"])
LLM_TOKENS = Histogram(
    "rag_llm_tokens", "Tokens per Ollama call", ["kind"],
    buckets=(16, 32, 64, 128, 256, 512, 1024, 2048, 4096))
LLM_FALLBACK = Counter("rag_llm_fallback_total",
                       "Times the dummy LLM was used because Ollama was unreachable")


def record_request(route: str, status: str, duration_s: float) -> None:
    REQUEST_DURATION.labels(route=route).observe(duration_s)
    REQUESTS_TOTAL.labels(route=route, status=status).inc()


def record_steps(steps: list[dict]) -> None:
    for s in steps:
        NODE_DURATION.labels(node=s["node"]).observe(s["duration_ms"] / 1000.0)
        if s["node"] == "validate" and "retry=True" in str(s.get("summary", "")):
            VALIDATE_RETRIES.inc()


def record_context(rag_context: str) -> None:
    CONTEXT_WORDS.observe(len(rag_context.split()))


@contextmanager
def time_subgraph_node(node: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        SUBGRAPH_NODE_DURATION.labels(node=node).observe(time.perf_counter() - start)


@contextmanager
def time_llm_call(op: str, provider: str):
    LLM_CALLS.labels(op=op, provider=provider).inc()
    start = time.perf_counter()
    try:
        yield
    finally:
        LLM_DURATION.labels(op=op).observe(time.perf_counter() - start)


def record_llm_tokens(prompt_tokens: int | None, output_tokens: int | None) -> None:
    if prompt_tokens:
        LLM_TOKENS.labels(kind="prompt").observe(prompt_tokens)
    if output_tokens:
        LLM_TOKENS.labels(kind="output").observe(output_tokens)


def metrics_asgi_app():
    return make_asgi_app()
