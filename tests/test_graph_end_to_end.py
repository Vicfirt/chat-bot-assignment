import pytest

from app.graph.main_graph import run_agent


@pytest.fixture(autouse=True)
def _seed_index(tmp_path, monkeypatch):
    from app.rag import retriever as rmod

    r = rmod.ChromaRetriever(chroma_dir=str(tmp_path / "c"), collection="tst")
    r.add_chunks([
        {"chunk_id": "pub501-29-0",
         "text": "For 2024 the standard deduction is 14,600 for single filers.",
         "pub": "Pub. 501", "section": "Standard Deduction", "page": 29,
         "source_url": "http://irs/p501", "tax_year": 2024},
        {"chunk_id": "pub17-1-0",
         "text": "Filing status determines your standard deduction and tax rates.",
         "pub": "Pub. 17", "section": "Filing Status", "page": 1,
         "source_url": "http://irs/p17", "tax_year": 2024},
    ])
    monkeypatch.setattr(rmod, "get_retriever", lambda: r)
    import app.rag.subgraph as sg
    sg._compiled = None


def test_rag_only_flow_has_citations_and_steps():
    state = run_agent("What is the standard deduction for single filers?", [])
    assert state["route"] in {"rag_only", "rag_plus_calc"}
    assert state["citations"]
    assert state["final_answer"]
    nodes = [s["node"] for s in state["steps"]]
    assert "triage" in nodes and "retrieve" in nodes and "synthesize" in nodes and "validate" in nodes


def test_calc_flow_produces_calc_result():
    state = run_agent(
        "How much federal tax do I owe on $85,000 filing single with 0 dependents in 2024?", []
    )
    assert state["calc_result"] and state["calc_result"].get("total_tax", 0) > 0


def test_rag_plus_calc_fans_out_retrieve_and_calculate():
    state = run_agent(
        "What is my federal tax after the standard deduction on $85,000, single, 0 dependents, 2024?",
        [],
    )
    assert state["route"] == "rag_plus_calc"
    nodes = [s["node"] for s in state["steps"]]
    # plan fans out to both independent subtasks, which rejoin at synthesize
    assert "retrieve" in nodes and "calculate" in nodes
    assert nodes.index("plan") < nodes.index("retrieve")
    assert nodes.index("plan") < nodes.index("calculate")
    assert max(nodes.index("retrieve"), nodes.index("calculate")) < nodes.index("synthesize")
    assert state["citations"] and state["calc_result"]["total_tax"] > 0


def test_rag_only_skips_calculate():
    state = run_agent("What is the standard deduction for single filers?", [])
    if state["route"] == "rag_only":
        assert "calculate" not in [s["node"] for s in state["steps"]]
        assert state.get("calc_result") is None


def test_out_of_scope_short_circuits():
    state = run_agent("What is the capital of France?", [])
    assert state["route"] == "out_of_scope"
    assert "scope" in state["final_answer"].lower()


def test_validate_failure_retries_retrieve_once(monkeypatch):
    import app.graph.nodes.retrieve as rn

    calls = {"n": 0}

    def flaky_tool(question, history, tax_profile=None):
        calls["n"] += 1
        if calls["n"] == 1:                       # first pass: nothing found -> retry
            return {"rag_context": "", "citations": [], "funnel": {}}
        return {
            "rag_context": "The 2025 standard deduction for single filers is $15,750.",
            "citations": [{"pub": "Pub. 17", "section": "Standard Deduction", "page": 97,
                           "source_url": "u", "quote": "standard deduction $15,750"}],
            "funnel": {"candidates": 3, "reranked": 1, "kept": 1},
        }

    monkeypatch.setattr(rn, "retriever_tool", flaky_tool)
    state = run_agent("What is the standard deduction for single filers?", [])

    assert calls["n"] == 2                        # retried exactly once
    assert [s["node"] for s in state["steps"]].count("retrieve") == 2
    assert state["citations"] and state["validation"]["ok"] is True
