# Agentic RAG Tax Chatbot

An Agentic RAG chatbot (LangGraph) that answers U.S. federal individual income
tax questions grounded in IRS publications (Pub. 17, 501, 505) and performs
deterministic federal tax estimates.

## Problem and objectives

Individual tax rules are high-stakes and easy to misread. This assistant:

- answers rule-lookup questions with a citation to the exact publication,
  section, and page;
- when asked, estimates federal income tax from a filing status and income
  using a deterministic calculator (no LLM arithmetic);
- routes every question autonomously (rule lookup, calculation, both, or
  out-of-scope) instead of relying on the user to pick a mode;
- runs fully offline and reproducibly in `LLM_MODE=dummy`, and against a real
  local model (Ollama) in `LLM_MODE=ollama`.

See `docs/superpowers/specs/2026-09-07-agentic-rag-tax-chatbot-design.md` for the
full rationale and non-goals.

## Architecture

Three core services: **Streamlit UI** -> **FastAPI** (hosts the LangGraph app +
embedded Chroma) -> **Ollama** (local LLM; `LLM_MODE=dummy` bypasses it).

### Main graph (6 nodes)

```
                           /--rag_only------------> retrieve --------------\
triage --route_after_triage ---rag_plus_calc--> plan --> retrieve --> calculate --> synthesize --> validate --(retry->retrieve | end)
                           \--needs_calc-------> plan --> calculate ------/                              ^
                           \--out_of_scope--------------------------------------------> synthesize ------/
```

- `rag_only` skips `plan`; `needs_calc` skips `retrieve`; `rag_plus_calc` runs
  both. `calculate` is a no-op passthrough on routes that don't need it.
- `synthesize -> validate` is unconditional; `validate` either ends or sends one
  retry back to `retrieve`.

Compiled node set (verified): `triage, plan, retrieve, calculate, synthesize,
validate` (plus `__start__` / `__end__`).

- **triage** — LLM classifies the question into `rag_only` / `needs_calc` /
  `rag_plus_calc` / `out_of_scope` (autonomous routing).
- **plan** — decomposes the question into subtasks and extracts a
  `tax_profile` (filing status, income, dependents, tax year). Skipped on
  `rag_only` / `out_of_scope`.
- **retrieve** — invokes the modular **RAG subgraph**. Tool #1 (retrieval).
  Skipped on `needs_calc` / `out_of_scope`.
- **calculate** — deterministic `estimate_tax(...)`. Tool #2 (non-retrieval);
  a no-op passthrough on the `rag_only` / `out_of_scope` routes.
- **synthesize** — LLM composes a cited answer from context and/or calc result.
- **validate** — checks citations and numeric consistency; on failure it sends
  exactly one retry back to `retrieve` (re-running retrieval -> calculate ->
  synthesize), then ends regardless of the second result.

### RAG subgraph (separate compiled graph, `app/rag/subgraph.py`)

```
expand_query --> retrieve_candidates --> rerank --> grade_docs --(loop: broaden & retry | continue)--> assemble_context
```

Compiled node set (verified): `expand_query, retrieve_candidates, rerank,
grade_docs, assemble_context` (plus `__start__` / `__end__`). Each node is one
named RAG subsystem: query expansion (LLM + deterministic domain hints),
candidate retrieval (dense + BM25 + RRF fusion), cross-encoder reranking (with
a table-first boost for amount questions), relevance grading, and context
assembly with citations.

### Tools

| Tool | File | Kind |
|------|------|------|
| Retriever (RAG subgraph) | `app/tools/retriever_tool.py` | retrieval |
| Federal tax estimator | `app/tools/tax_calculator.py` | non-retrieval, deterministic |

### Design rationale (highlights; full detail in spec §17)

- **Approach A (fixed agentic graph) over a ReAct agent** — small local models
  tool-call unreliably; an explicit graph makes routing and retries auditable.
- **Local `sentence-transformers` embeddings + embedded Chroma** — no API key,
  deterministic, metadata-aware, runs in the same process as the API.
- **Structure-aware chunking** — tax rules are hierarchical and citations need
  `section` + `page`, so chunks carry publication / section / page metadata.
- **Pluggable `ollama` / `dummy` LLM** — reconciles "real local model" with
  "reproducible and offline for graders". `dummy` is the default for tests,
  eval, and load tests.
- **Deterministic calculator, never LLM arithmetic** — tax math must be exact
  and testable.

## Evaluation and performance — results

Both artifacts below are generated, not hand-written:
`LLM_MODE=dummy python -m eval.run_eval` and
`LLM_MODE=dummy python -m loadtest.run_load`.

### Functional evaluation — `docs/eval-results.md` (+ `.json`)

16 questions spanning all four routes. `run_eval` writes both a markdown table
and a machine-readable `docs/eval-results.json`.

| Metric | `ollama` / `llama3.2:3b` |
|--------|--------|
| Route accuracy | see `docs/eval-results.md` |
| Retrieval hit rate (cited pub matches expected) | " |
| Citation rate | " |
| Numeric accuracy (calc questions) | " |
| Keyword hit rate (brittle substring proxy) | " |

Route accuracy and `tax_profile` extraction depend on the LLM, so run under
`LLM_MODE=ollama`; retrieval hit rate and citation rate are largely independent
of generation quality.

### Retrieval quality — `## Retrieval quality` section of the same file

Judged at `(publication, page)` granularity against the `relevant_pages` labels
in `eval/questions.yaml`, by invoking the RAG subgraph directly and reading its
`raw_hits` (post RRF fusion) → `reranked_hits` (post cross-encoder) →
`graded_hits` (kept for the prompt).

| Metric | Meaning |
|--------|---------|
| `precision_at_k`, `recall_at_k` | of the reranked top-k (k = `search_k`) |
| `mrr` | mean reciprocal rank of the first relevant page |
| `context_precision` | fraction of the assembled context that is relevant |
| `mrr_fused` vs `mrr_reranked`, `rerank_mrr_lift` | the ranking gain the `rerank` node adds over pure RRF |
| `hit_rate_fused` vs `hit_rate_reranked`, `rerank_hit_lift` | same, as a top-k hit rate |

### Load test — `docs/loadtest-results.md`

100 requests, `LLM_MODE=dummy`, embedded Chroma warm, run at **concurrency 1**
(see the caveat below).

| p50 | p90 | p95 | p99 | mean | max | throughput |
|-----|-----|-----|-----|------|-----|------------|
| 37.4 ms | 67.3 ms | 72.1 ms | 150.2 ms | 101.0 ms | 6700.6 ms | 9.9 req/s |

`max` is the first (cold) request that loads the HNSW index into memory. A
latency histogram is written to `docs/loadtest-latency.png` locally (gitignored).

**Bottleneck node: `retrieve`** — mean 108.8 ms per request, essentially the
entire request budget (`triage`, `plan`, `calculate`, `synthesize`, `validate`
are ~0 ms in `dummy` mode). In `dummy` mode this cost is vector search over the
embedded Chroma index; in `ollama` mode `synthesize` (LLM generation on CPU)
typically dominates instead. Recommended optimizations:

1. Cut LLM calls on the retrieval path: replace the `grade_docs` LLM grader
   with a score threshold and merge query expansion into a single call
   (~2 fewer LLM round-trips per request in `ollama` mode).
2. Add a semantic response cache keyed on normalized question + route, and skip
   the `validate` -> `retrieve` retry when the draft already has citations and
   the calc number.

**Known issue — concurrency:** the API is served by a single `uvicorn` process
with a synchronous `/chat` handler, so concurrent requests run on the thread
pool and share one embedded-Chroma client. At concurrency >= 2 the retrieval
path crashes the worker (native crash, no Python traceback; the load test then
reports connection errors). The load-test artifact is therefore captured at
concurrency 1. Fixing this is the top reliability item: give each worker its own
Chroma client / serialize retrieval, or run the model server (or a retrieval
service) out of process and scale the API with multiple worker processes.

## Install and run

Python 3.11. No paid APIs. `LLM_MODE=dummy` is the default for offline,
reproducible runs; `LLM_MODE=ollama` needs the model pulled first.

### Prerequisites

Docker + Docker Compose. ~4 GB free RAM for the 3B model (not needed in
`dummy` mode).

### First run (Docker)

```bash
cp .env.example .env
make ingest                                            # build the vector index (needs internet)
docker compose exec ollama ollama pull llama3.2:3b     # or set LLM_MODE=dummy in .env
docker compose up --build                              # UI on :8501, API on :8000
```

`make ingest` downloads IRS Pub. 17 / 501 / 505 and builds the embedded Chroma
index under `data/chroma/`. It needs internet once; the SHA-256 of each PDF is
pinned in `app/ingest/sources.yaml`.

### Dummy mode (no model download, fully offline)

```bash
LLM_MODE=dummy docker compose up --build api ui
```

### Local (no Docker)

```bash
make install
python -m app.ingest.build_index                       # needs internet, one time
LLM_MODE=dummy uvicorn app.api.main:app --port 8000
LLM_MODE=dummy streamlit run app/ui/streamlit_app.py   # separate shell
```

### Observability (optional)

```bash
docker compose --profile observability up --build
# Grafana http://localhost:3001 (anon)  |  Prometheus :9090
# Pushgateway :9091  |  Langfuse :3000 (set LANGFUSE_ENABLED=true in .env)
```

The API exposes Prometheus metrics at `/metrics`; the provisioned Grafana
dashboard ("Agentic RAG Tax Chatbot") shows:

| Group | Metrics |
|-------|---------|
| Request | `rag_request_duration_seconds` (p95 by route), `rag_requests_total` (throughput, error ratio) |
| Graph | `rag_node_duration_seconds` (6 main nodes), `rag_subgraph_node_duration_seconds` (5 RAG nodes), `rag_validate_retries_total` |
| LLM | `rag_llm_duration_seconds` (p95 by op), `rag_llm_calls_total` (by op), `rag_llm_tokens` (prompt/output), `rag_llm_fallback_total` (Ollama-unreachable → dummy) |
| Retrieval | `rag_context_words`, `rag_retrieval_chunks` |
| Offline eval | `rag_eval_*` gauges — `run_eval` pushes route accuracy, retrieval hit rate, Precision@k, Recall@k, MRR and rerank lift to the pushgateway when `PROM_PUSHGATEWAY` is set |

```bash
PROM_PUSHGATEWAY=localhost:9091 OLLAMA_BASE_URL=http://localhost:11434 \
  LLM_MODE=ollama python -m eval.run_eval      # results also land in Grafana
```

### Tests / eval / load test

```bash
make test                                              # LLM_MODE=dummy python -m pytest -q
LLM_MODE=dummy python -m eval.run_eval                  # writes docs/eval-results.md
LLM_MODE=dummy python -m loadtest.run_load --api-url http://localhost:8000 --n 100 --concurrency 1
# API must be running; see the concurrency caveat above before raising --concurrency
```
