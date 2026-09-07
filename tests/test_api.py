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

    monkeypatch.setattr(api, "run_agent", fake_run_agent)
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_chat_returns_answer_and_trace(client):
    r = client.post("/chat", json={"question": "standard deduction?"})
    assert r.status_code == 200
    body = r.json()
    assert "14,600" in body["answer"]
    assert body["route"] == "rag_only"
    assert body["timings"]["per_node_ms"]["retrieve"] == 5.0
    assert body["citations"][0]["pub"] == "Pub. 501"


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"rag_requests_total" in r.content
