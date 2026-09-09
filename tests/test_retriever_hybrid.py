import pytest

from app.rag.retriever import Chunk, ChromaRetriever, HybridRetriever


def _c(cid, text, *, block_type="prose", pub="Pub. 501", page=1, score=0.0):
    return Chunk(cid, text, pub, "S", page, "u", 2024, score, block_type)


@pytest.fixture()
def seeded(tmp_path):
    r = ChromaRetriever(chroma_dir=str(tmp_path / "c"), collection="hyb")
    r.add_chunks([
        {"chunk_id": "sd-tab", "block_type": "table",
         "text": "[Table - Standard Deduction (Pub. 501 p.29)]\nSingle | $14,600\n"
                 "Married filing jointly | $29,200\nHead of household | $21,900",
         "pub": "Pub. 501", "section": "Standard Deduction", "page": 29,
         "source_url": "u", "tax_year": 2025},
        {"chunk_id": "sd-prose", "block_type": "prose",
         "text": "The standard deduction reduces the income you are taxed on. Most filers "
                 "take it instead of itemizing.",
         "pub": "Pub. 501", "section": "Standard Deduction", "page": 22,
         "source_url": "u", "tax_year": 2025},
        {"chunk_id": "est-prose", "block_type": "prose",
         "text": "You may owe estimated tax if you have income not subject to withholding.",
         "pub": "Pub. 505", "section": "Estimated Tax", "page": 5,
         "source_url": "u", "tax_year": 2025},
        {"chunk_id": "old-tab", "block_type": "table",
         "text": "[Table - Standard Deduction (Pub. 501 p.29)]\nSingle | $13,850",
         "pub": "Pub. 501", "section": "Standard Deduction", "page": 29,
         "source_url": "u", "tax_year": 2024},
    ])
    return HybridRetriever(dense=r)


def test_bm25_matches_exact_terms(seeded):
    hits = seeded._bm25_search("estimated tax withholding", k=5)
    assert hits and hits[0].chunk_id == "est-prose"


def test_rrf_fuses_and_dedupes():
    hr = HybridRetriever.__new__(HybridRetriever)
    a = [_c("x", "a"), _c("y", "b"), _c("z", "c")]
    b = [_c("y", "b"), _c("x", "a")]
    fused = HybridRetriever._rrf(hr, [a, b], k=3)
    ids = [c.chunk_id for c in fused]
    assert set(ids) == {"x", "y", "z"}
    assert ids[0] in {"x", "y"}                 # appear in both lists -> ranked above z
    assert ids[-1] == "z"


def test_tax_year_filter_excludes_other_years(seeded, monkeypatch):
    monkeypatch.setenv("RERANK_ENABLED", "false")
    monkeypatch.setenv("FILTER_TAX_YEAR", "true")
    from app.config import get_settings

    get_settings.cache_clear()
    hits = seeded.search("standard deduction amount for single filers", k=10)
    get_settings.cache_clear()
    assert hits
    assert all(h.tax_year == 2025 for h in hits)
    assert "old-tab" not in {h.chunk_id for h in hits}


def test_table_boost_puts_tables_first_for_amount_queries(seeded, monkeypatch):
    monkeypatch.setenv("RERANK_ENABLED", "false")
    monkeypatch.setenv("BOOST_TABLES_FOR_AMOUNT_QUERIES", "true")
    from app.config import get_settings

    get_settings.cache_clear()
    hits = seeded.search("how much is the standard deduction?", k=5)
    get_settings.cache_clear()
    assert hits and hits[0].block_type == "table"


def test_amount_query_keeps_tables_the_cross_encoder_scores_negative(seeded, monkeypatch):
    monkeypatch.setenv("RERANK_ENABLED", "true")
    monkeypatch.setenv("BOOST_TABLES_FOR_AMOUNT_QUERIES", "true")
    from app.config import get_settings

    get_settings.cache_clear()

    class FakeCE:
        # a prose-trained CE scores a bare number table well below zero
        def predict(self, pairs):
            return [-4.0 if "|" in t else 2.0 for _q, t in pairs]

    seeded._reranker = FakeCE()
    out = seeded.rerank("how much is the 2025 standard deduction?", [
        _c("prose", "Some paragraph about the standard deduction rules."),
        _c("tab", "Single | $15,750\nMFJ | $31,500", block_type="table"),
    ], k=5)
    get_settings.cache_clear()
    # the table survives (would be dropped by the >= grade_min_score floor)
    assert out[0].chunk_id == "tab"


def test_rerank_reorders_by_cross_encoder(seeded, monkeypatch):
    monkeypatch.setenv("RERANK_ENABLED", "true")
    monkeypatch.setenv("BOOST_TABLES_FOR_AMOUNT_QUERIES", "false")
    from app.config import get_settings

    get_settings.cache_clear()

    class FakeCE:
        def predict(self, pairs):
            # score highest for whichever candidate text contains "29,200"
            return [10.0 if "29,200" in t else -1.0 for _q, t in pairs]

    seeded._reranker = FakeCE()
    out = seeded.rerank("married filing jointly deduction", [
        _c("a", "prose about deductions"), _c("b", "Married filing jointly | $29,200"),
    ], k=2)
    get_settings.cache_clear()
    assert out[0].chunk_id == "b"
