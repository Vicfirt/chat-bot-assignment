import pytest

from app.rag import cache
from app.rag.cache import LRU, normalize_question


def test_lru_evicts_oldest_and_counts():
    c = LRU(2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1          # a now most-recent
    c.put("c", 3)                    # evicts b
    assert c.get("b") is None
    assert c.get("c") == 3
    assert c.hits == 2 and c.misses == 1


def test_normalize_question_collapses_ws_and_case():
    assert normalize_question("  What   is\tthe   Standard Deduction? ") == \
        "what is the standard deduction?"


def test_embedding_cache_avoids_re_encoding(tmp_path, monkeypatch):
    from app.rag.retriever import ChromaRetriever

    r = ChromaRetriever(chroma_dir=str(tmp_path / "c"), collection="emb")
    calls = {"n": 0}

    def fake_encode(texts):
        calls["n"] += 1
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(r, "_encode", fake_encode)
    assert r._embed(["standard deduction"]) == [[0.1, 0.2, 0.3]]
    assert r._embed(["standard deduction"]) == [[0.1, 0.2, 0.3]]
    assert calls["n"] == 1                     # second call served from cache


def test_index_fingerprint_changes_with_settings(monkeypatch):
    fp1 = cache.index_fingerprint()
    monkeypatch.setenv("RERANK_TOP_N", "99")
    from app.config import get_settings

    get_settings.cache_clear()
    assert cache.index_fingerprint() != fp1


def test_run_rag_second_call_is_cached(tmp_path, monkeypatch):
    from app.rag import retriever as rmod
    import app.rag.subgraph as sg

    r = rmod.ChromaRetriever(chroma_dir=str(tmp_path / "c"), collection="tcache")
    r.add_chunks([{"chunk_id": "p1", "text": "standard deduction 14,600 single",
                   "pub": "Pub. 501", "section": "SD", "page": 1,
                   "source_url": "u", "tax_year": 2025}])
    monkeypatch.setattr(rmod, "get_retriever", lambda: r)

    calls = {"n": 0}
    real_build = sg.build_rag_subgraph

    def counting_build():
        calls["n"] += 1
        return real_build()

    monkeypatch.setattr(sg, "build_rag_subgraph", counting_build)
    sg._compiled = None
    cache.clear_all()

    a = sg.run_rag("what is the standard deduction?", [])
    b = sg.run_rag("What is the Standard Deduction?", [])   # same after normalize
    assert a == b
    assert cache._rag_cache.hits == 1
