from app.rag.nodes.expand_query import expand_query
from app.rag.nodes.rerank import rerank
from app.rag.nodes.retrieve_candidates import retrieve_candidates


def test_expand_query_includes_original_and_counts_round():
    out = expand_query({"question": "standard deduction for single", "chat_history": [],
                        "rounds": 0})
    assert out["queries"][0] == "standard deduction for single"
    assert out["rounds"] == 1
    assert len(out["queries"]) <= 5


def test_expand_query_adds_domain_hint_for_brackets():
    out = expand_query({"question": "what are the tax brackets for a single filer",
                        "chat_history": [], "rounds": 0})
    assert any("Tax Rate Schedules" in q for q in out["queries"])


def test_expand_query_condenses_followup_against_history(monkeypatch):
    from app.rag.nodes import expand_query as eq

    class _Stub:
        def complete(self, prompt, *, system=None, max_tokens=512):
            if "Follow-up:" in prompt:
                return "What is the standard deduction for married filing jointly?"
            return ""   # no extra expansion lines

    monkeypatch.setattr(eq, "get_llm", lambda: _Stub())
    out = eq.expand_query({
        "question": "what about married filing jointly?",
        "chat_history": [
            {"role": "user", "content": "What is the standard deduction for a single filer?"},
            {"role": "assistant", "content": "It is $15,750 for 2025."},
        ],
        "rounds": 0,
    })
    assert out["queries"][0] == "What is the standard deduction for married filing jointly?"


def test_expand_query_no_history_uses_question_verbatim():
    out = expand_query({"question": "qualifying child tests", "chat_history": [], "rounds": 0})
    assert out["queries"][0] == "qualifying child tests"


def test_retrieve_candidates_unions_and_dedupes(monkeypatch):
    from app.rag import retriever as rmod

    class FakeRetriever:
        def search_candidates(self, query, k):
            return [rmod.Chunk("a-1-0", "txt A", "Pub. 501", "S", 1, "u", 2024, 0.4 if "x" in query else 0.9),
                    rmod.Chunk("b-1-0", "txt B", "Pub. 505", "S", 1, "u", 2024, 0.2)]

    monkeypatch.setattr(rmod, "get_retriever", lambda: FakeRetriever())
    out = retrieve_candidates({"queries": ["q1", "x q2"], "chat_history": [], "question": "q"})
    hits = {h["chunk_id"]: h for h in out["raw_hits"]}
    assert set(hits) == {"a-1-0", "b-1-0"}
    assert hits["a-1-0"]["score"] == 0.9


def test_rerank_node_reorders_and_passes_through_without_reranker(monkeypatch):
    from app.rag import retriever as rmod

    raw = [{"chunk_id": "a", "text": "prose", "pub": "Pub. 501", "section": "S", "page": 1,
            "source_url": "u", "tax_year": 2025, "score": 0.9, "block_type": "prose"},
           {"chunk_id": "b", "text": "Married filing jointly | $29,200", "pub": "Pub. 501",
            "section": "S", "page": 2, "source_url": "u", "tax_year": 2025, "score": 0.5,
            "block_type": "table"}]

    class WithReranker:
        def rerank(self, question, chunks, k):
            return list(reversed(chunks))

    monkeypatch.setattr(rmod, "get_retriever", lambda: WithReranker())
    out = rerank({"question": "mfj deduction", "raw_hits": raw})
    assert [h["chunk_id"] for h in out["reranked_hits"]] == ["b", "a"]

    class NoReranker:
        pass

    monkeypatch.setattr(rmod, "get_retriever", lambda: NoReranker())
    out = rerank({"question": "q", "raw_hits": raw})
    assert [h["chunk_id"] for h in out["reranked_hits"]] == ["a", "b"]


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
