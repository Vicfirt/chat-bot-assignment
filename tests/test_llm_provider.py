import app.llm.provider as provider
from app.llm.provider import DummyLLM, get_llm


def test_get_llm_returns_dummy_in_dummy_mode():
    assert isinstance(get_llm(), DummyLLM)


def test_get_llm_falls_back_to_dummy_when_ollama_unreachable(monkeypatch):
    # ollama mode, but the probe fails -> fallback to the dummy LLM
    import sys
    import types

    fake_ollama = types.ModuleType("ollama")

    def _boom(*_a, **_kw):
        raise ConnectionError("no ollama in tests")

    fake_ollama.Client = _boom
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)
    monkeypatch.setattr(provider, "_ollama_ready", None, raising=False)
    monkeypatch.setenv("LLM_MODE", "ollama")
    monkeypatch.setenv("LLM_FALLBACK_DUMMY", "true")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        assert isinstance(get_llm(), DummyLLM)
    finally:
        get_settings.cache_clear()
        provider._ollama_ready = None


def test_dummy_classify_returns_a_label():
    llm = DummyLLM()
    out = llm.classify("What is the standard deduction?", ["rag_only", "needs_calc"])
    assert out in {"rag_only", "needs_calc"}


def test_dummy_classify_detects_calc_marker():
    llm = DummyLLM()
    out = llm.classify(
        "how much tax do I owe on 85000 dollars, filing single",
        ["rag_only", "needs_calc", "rag_plus_calc", "out_of_scope"],
    )
    assert out in {"needs_calc", "rag_plus_calc"}


def test_dummy_complete_is_deterministic():
    llm = DummyLLM()
    assert llm.complete("hello world") == llm.complete("hello world")
