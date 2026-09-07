import pytest

from app.rag.subgraph import run_rag


@pytest.fixture(autouse=True)
def _seed_index(tmp_path, monkeypatch):
    from app.rag import retriever as rmod

    r = rmod.ChromaRetriever(chroma_dir=str(tmp_path / "c"), collection="tst")
    r.add_chunks([
        {"chunk_id": "pub501-29-0",
         "text": "For 2024 the standard deduction is 14,600 for single and 29,200 for married filing jointly.",
         "pub": "Pub. 501", "section": "Standard Deduction", "page": 29,
         "source_url": "http://irs/p501", "tax_year": 2024},
        {"chunk_id": "pub505-5-0",
         "text": "Estimated tax is used to pay tax on income not subject to withholding.",
         "pub": "Pub. 505", "section": "Estimated Tax", "page": 5,
         "source_url": "http://irs/p505", "tax_year": 2024},
    ])
    monkeypatch.setattr(rmod, "get_retriever", lambda: r)
    import app.rag.subgraph as sg
    sg._compiled = None


def test_run_rag_returns_context_and_citations():
    out = run_rag("what is the standard deduction for single filers?", [])
    assert "14,600" in out["rag_context"]
    assert out["citations"]
    assert out["citations"][0]["pub"] == "Pub. 501"
    assert out["citations"][0]["page"] == 29
