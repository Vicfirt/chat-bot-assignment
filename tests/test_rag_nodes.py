from app.rag.nodes.expand_query import expand_query
from app.rag.nodes.vector_search import vector_search


def test_expand_query_includes_original_and_counts_round():
    out = expand_query({"question": "standard deduction for single", "chat_history": [],
                        "rounds": 0})
    assert out["queries"][0] == "standard deduction for single"
    assert out["rounds"] == 1
    assert len(out["queries"]) <= 3


def test_vector_search_unions_and_dedupes(monkeypatch):
    from app.rag import retriever as rmod

    class FakeRetriever:
        def search(self, query, k):
            return [rmod.Chunk("a-1-0", "txt A", "Pub. 501", "S", 1, "u", 2024, 0.4 if "x" in query else 0.9),
                    rmod.Chunk("b-1-0", "txt B", "Pub. 505", "S", 1, "u", 2024, 0.2)]

    monkeypatch.setattr(rmod, "get_retriever", lambda: FakeRetriever())
    out = vector_search({"queries": ["q1", "x q2"], "chat_history": [], "question": "q"})
    hits = {h["chunk_id"]: h for h in out["raw_hits"]}
    assert set(hits) == {"a-1-0", "b-1-0"}
    assert hits["a-1-0"]["score"] == 0.9


def test_grade_docs_keeps_min_docs_even_below_threshold():
    from app.rag.nodes.grade_docs import grade_docs, route_after_grade

    state = {"raw_hits": [{"score": 0.1, "text": "a"}, {"score": 0.05, "text": "b"},
                          {"score": 0.02, "text": "c"}], "rounds": 2}
    out = grade_docs(state)
    assert len(out["graded_hits"]) == 3
    assert route_after_grade({**state, **out}) == "assemble"


def test_assemble_context_respects_budget_and_builds_citations():
    from app.rag.nodes.assemble_context import assemble_context

    hits = [{"score": 0.9, "text": "word " * 50, "pub": "Pub. 501",
             "section": "SD", "page": 3, "source_url": "u"},
            {"score": 0.8, "text": "word " * 5000, "pub": "Pub. 17",
             "section": "X", "page": 9, "source_url": "u"}]
    out = assemble_context({"graded_hits": hits})
    assert out["citations"][0]["pub"] == "Pub. 501"
    assert "(p.3)" in out["rag_context"]
