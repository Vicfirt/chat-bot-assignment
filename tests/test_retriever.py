from app.rag.retriever import Chunk, ChromaRetriever


def _chunks():
    return [
        {"chunk_id": "pub501-1-0",
         "text": "The standard deduction for single filers in 2024 is 14,600 dollars.",
         "pub": "Pub. 501", "section": "Standard Deduction", "page": 1,
         "source_url": "http://x", "tax_year": 2024},
        {"chunk_id": "pub505-3-0",
         "text": "You may need to pay estimated tax if you have income not subject to withholding.",
         "pub": "Pub. 505", "section": "Estimated Tax", "page": 3,
         "source_url": "http://y", "tax_year": 2024},
    ]


def test_add_and_search_returns_relevant_chunk(tmp_path):
    r = ChromaRetriever(chroma_dir=str(tmp_path / "c"), collection="test")
    r.add_chunks(_chunks())
    assert r.count() == 2
    hits = r.search("what is the standard deduction for a single person", k=2)
    assert hits and isinstance(hits[0], Chunk)
    assert hits[0].chunk_id == "pub501-1-0"
    assert hits[0].score >= hits[-1].score
