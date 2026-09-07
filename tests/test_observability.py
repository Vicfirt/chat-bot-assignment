from app.observability.metrics import record_request, record_steps
from app.observability.tracing import get_langfuse_callbacks


def test_record_steps_and_request_do_not_raise():
    record_steps([{"node": "triage", "duration_ms": 3.2, "summary": "route=rag_only"},
                  {"node": "validate", "duration_ms": 1.0, "summary": "retry=True reasons=['x']"}])
    record_request("rag_only", "ok", 0.5)


def test_langfuse_disabled_returns_empty(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    from app.config import get_settings

    get_settings.cache_clear()
    assert get_langfuse_callbacks() == []
