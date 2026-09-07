from __future__ import annotations

from prometheus_client import Counter, Histogram, make_asgi_app

REQUEST_DURATION = Histogram("rag_request_duration_seconds",
                             "End-to-end /chat duration", ["route"])
NODE_DURATION = Histogram("rag_node_duration_seconds", "Per-node duration", ["node"])
REQUESTS_TOTAL = Counter("rag_requests_total", "Chat requests", ["route", "status"])
VALIDATE_RETRIES = Counter("rag_validate_retries_total", "Validate-triggered retries")
RETRIEVAL_CHUNKS = Histogram("rag_retrieval_chunks", "Citations returned per request")


def record_request(route: str, status: str, duration_s: float) -> None:
    REQUEST_DURATION.labels(route=route).observe(duration_s)
    REQUESTS_TOTAL.labels(route=route, status=status).inc()


def record_steps(steps: list[dict]) -> None:
    for s in steps:
        NODE_DURATION.labels(node=s["node"]).observe(s["duration_ms"] / 1000.0)
        if s["node"] == "validate" and "retry=True" in str(s.get("summary", "")):
            VALIDATE_RETRIES.inc()


def metrics_asgi_app():
    return make_asgi_app()
