import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    import app.api.main as api

    def fake_run_agent(question, chat_history=None, callbacks=None):
        return {"route": "rag_only", "final_answer": "The standard deduction is 14,600. [Pub. 501 p.29]",
                "citations": [{"pub": "Pub. 501", "page": 29}],
                "validation": {"low_confidence": False},
                "steps": [{"node": "triage", "duration_ms": 2.0, "summary": "route=rag_only"},
                          {"node": "retrieve", "duration_ms": 5.0, "summary": "1 citations"}]}

    def fake_run_agent_stream(question, chat_history=None, callbacks=None):
        yield {"triage": {"route": "rag_only",
                          "steps": [{"node": "triage", "duration_ms": 2.0, "summary": "route=rag_only"}]}}
        yield {"retrieve": {"citations": [{"pub": "Pub. 501", "page": 29}],
                            "steps": [{"node": "retrieve", "duration_ms": 5.0, "summary": "1 citations"}]}}
        yield {"synthesize": {"final_answer": "The standard deduction is 14,600. [Pub. 501 p.29]",
                              "steps": [{"node": "synthesize", "duration_ms": 1.0, "summary": "done"}]}}
        yield {"validate": {"validation": {"low_confidence": False},
                            "steps": [{"node": "validate", "duration_ms": 0.5, "summary": "ok"}]}}

    monkeypatch.setattr(api, "run_agent", fake_run_agent)
    monkeypatch.setattr(api, "run_agent_stream", fake_run_agent_stream)
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "degraded"}   # degraded when no index is built
    assert "index_chunks" in body


def test_chat_returns_answer_and_trace(client):
    r = client.post("/chat", json={"question": "standard deduction?"})
    assert r.status_code == 200
    body = r.json()
    assert "14,600" in body["answer"]
    assert body["route"] == "rag_only"
    assert body["timings"]["per_node_ms"]["retrieve"] == 5.0
    assert body["citations"][0]["pub"] == "Pub. 501"


def test_second_identical_chat_is_served_from_cache(monkeypatch):
    import app.api.main as api

    calls = {"n": 0}

    def counting_run_agent(question, chat_history=None, callbacks=None):
        calls["n"] += 1
        return {"route": "rag_only", "final_answer": "The standard deduction is 14,600. [Pub. 501 p.29]",
                "citations": [{"pub": "Pub. 501", "page": 29}],
                "validation": {"low_confidence": False},
                "steps": [{"node": "triage", "duration_ms": 2.0, "summary": "route=rag_only"}]}

    monkeypatch.setattr(api, "run_agent", counting_run_agent)
    c = TestClient(api.app)

    a = c.post("/chat", json={"question": "What is the standard deduction?"})
    b = c.post("/chat", json={"question": "  what IS the   Standard Deduction? "})  # same after normalise

    assert calls["n"] == 1                       # graph ran once
    assert a.json()["cached"] is False
    assert b.json()["cached"] is True
    assert b.json()["answer"] == a.json()["answer"]


def test_chat_with_history_is_not_cached(monkeypatch):
    import app.api.main as api

    calls = {"n": 0}

    def counting_run_agent(question, chat_history=None, callbacks=None):
        calls["n"] += 1
        return {"route": "rag_only", "final_answer": "answer [Pub. 17 p.1]",
                "citations": [{"pub": "Pub. 17", "page": 1}],
                "validation": {"low_confidence": False}, "steps": []}

    monkeypatch.setattr(api, "run_agent", counting_run_agent)
    c = TestClient(api.app)
    body = {"question": "and for married?", "chat_history": [{"role": "user", "content": "single deduction?"}]}
    c.post("/chat", json=body)
    c.post("/chat", json=body)
    assert calls["n"] == 2                       # history -> cache bypassed


def test_chat_stream_emits_steps_then_final(client):
    import json

    r = client.post("/chat/stream", json={"question": "standard deduction?"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    kinds = [e["type"] for e in events]
    assert kinds[:1] == ["step"] and kinds[-1] == "final"
    assert [e["node"] for e in events if e["type"] == "step"][0] == "triage"
    final = events[-1]
    assert "14,600" in final["answer"]
    assert final["route"] == "rag_only"
    assert final["citations"][0]["pub"] == "Pub. 501"


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"rag_requests_total" in r.content
