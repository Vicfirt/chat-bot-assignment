import httpx
import pytest

from loadtest.run_load import percentiles, run_load


def test_percentiles_basic():
    p = percentiles([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    assert p["p50"] == pytest.approx(55, abs=10)
    assert p["max"] == 100


@pytest.mark.asyncio
async def test_run_load_against_mock(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={
            "answer": "ok", "citations": [], "route": "rag_only", "timings": {},
            "steps": [{"node": "synthesize", "duration_ms": 40.0, "summary": ""},
                      {"node": "triage", "duration_ms": 2.0, "summary": ""}]})

    import loadtest.run_load as rl

    monkeypatch.setattr(rl, "_client_factory",
                        lambda *a, **k: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    result = await run_load("http://api:8000", n=12, concurrency=3)
    assert result["n"] == 12
    assert result["errors"] == 0
    assert result["per_node_ms"]["synthesize"] == pytest.approx(40.0)
