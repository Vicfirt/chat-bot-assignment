import httpx

from app.ui.streamlit_app import call_api


def test_call_api_posts_and_returns_json():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat"
        return httpx.Response(200, json={"answer": "ok", "citations": [], "route": "rag_only",
                                         "steps": [], "timings": {}, "low_confidence": False})

    transport = httpx.MockTransport(handler)
    out = call_api("q", [], "http://api:8000", client=httpx.Client(transport=transport))
    assert out["answer"] == "ok"
