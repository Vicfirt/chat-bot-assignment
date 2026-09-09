from concurrent.futures import ThreadPoolExecutor

from app.rag.retriever import ChromaRetriever, HybridRetriever


def _seeded(tmp_path):
    r = ChromaRetriever(chroma_dir=str(tmp_path / "c"), collection="conc")
    r.add_chunks([
        {"chunk_id": f"p{i}", "text": f"standard deduction row {i} for single filers 15,750",
         "pub": "Pub. 17", "section": "SD", "page": i, "source_url": "u", "tax_year": 2025}
        for i in range(30)
    ])
    return HybridRetriever(dense=r)


def test_parallel_search_does_not_crash_and_is_stable(tmp_path):
    hr = _seeded(tmp_path)
    baseline = [c.chunk_id for c in hr.search("standard deduction single", k=5)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(
            lambda _: [c.chunk_id for c in hr.search("standard deduction single", k=5)],
            range(40),
        ))

    assert all(r == baseline for r in results)      # serialised -> deterministic
