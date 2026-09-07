from app.llm.provider import DummyLLM, get_llm


def test_get_llm_returns_dummy_in_dummy_mode():
    assert isinstance(get_llm(), DummyLLM)


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
