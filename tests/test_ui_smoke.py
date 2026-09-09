import httpx

from app.ui.streamlit_app import call_api, stream_api


def test_call_api_posts_and_returns_json():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat"
        return httpx.Response(200, json={"answer": "ok", "citations": [], "route": "rag_only",
                                         "steps": [], "timings": {}, "low_confidence": False})

    transport = httpx.MockTransport(handler)
    out = call_api("q", [], "http://api:8000", client=httpx.Client(transport=transport))
    assert out["answer"] == "ok"


def test_stream_api_parses_sse_events():
    body = (
        'data: {"type": "step", "node": "triage", "duration_ms": 2.0, "summary": "route=rag_only"}\n\n'
        'data: {"type": "step", "node": "synthesize", "duration_ms": 1.0, "summary": "done"}\n\n'
        'data: {"type": "final", "answer": "hi", "citations": [], "route": "rag_only"}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat/stream"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    events = list(stream_api("q", [], "http://api:8000", client=httpx.Client(transport=transport)))
    assert [e["type"] for e in events] == ["step", "step", "final"]
    assert events[-1]["answer"] == "hi"
