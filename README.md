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

## Justification

- **Why the problem is relevant** — every US filer faces these rules yearly;
  the amounts (standard deduction, bracket thresholds) change annually and are
  easy to misquote, and a wrong figure has real financial and legal cost.
- **What user need it addresses** — a plain-language answer to "what is the
  rule" and "what would I owe", each tied to the exact IRS publication, section,
  and page so the user can verify rather than trust.
- **Why agentic RAG is the right fit** — the questions are not homogeneous: some
  need only a definition lookup, some only arithmetic from given numbers, some
  both, and some are off-topic. A router picks the sub-pipeline per question; a
  deterministic calculator handles the math so the LLM never does arithmetic;
  and a validate step re-grounds an unsupported answer with one retrieval retry.
  Plain single-shot RAG would run retrieval on calc-only questions, has no place
  for the calculator, and cannot self-correct a weakly-grounded draft.

## Architecture

Three core services: **Streamlit UI** -> **FastAPI** (hosts the LangGraph app +
embedded Chroma) -> **Ollama** (local LLM; `LLM_MODE=dummy` bypasses it).

The UI calls `POST /chat/stream` (server-sent events): each graph node emits its
`record_step` as it finishes, and the UI appends it to a live `st.status`
panel — so a multi-minute CPU run shows `triage ✓ → retrieve ✓ → rerank ✓ →
synthesize…` instead of one opaque spinner. `POST /chat` (blocking JSON) is
kept for programmatic use and the load test.

### Main graph (7 nodes)

```
triage --+-- rag_only ------------------------> retrieve ----------------+
         |                                                               |
         +-- needs_calc ----- plan ----------> calculate ----------------+--> synthesize --> guardrails --> validate --+
         |                                                               |                                    ^        |
         +-- rag_plus_calc -- plan --+--> retrieve --+  (parallel        |                                    |        |
         |                           +--> calculate -+   fan-out, both --+                          retry --> retrieve  |
         |                                                joins here)                                                   |
         +-- out_of_scope --------------------------------------------------> synthesize -------------------------------+
                                                                                                             end --> END
```

- `rag_only` runs only `retrieve`; `needs_calc` runs only `calculate`;
  `rag_plus_calc` fans `plan` out to **both, running concurrently**, and they
  rejoin at `synthesize` (LangGraph superstep barrier).
- `synthesize -> guardrails -> validate` is unconditional; `validate` either ends
  or sends one retry back to `retrieve` alone (the calc result is deterministic
  and persists in state).

Compiled node set (verified): `triage, plan, retrieve, calculate, synthesize,
guardrails, validate` (plus `__start__` / `__end__`).

**Decomposition into subtasks and independent execution.** A compound question
like "what's my tax after the standard deduction on $85k, single?" is broken
into subtasks that run as separate nodes with their own state slices: *look up
the governing rule* (`retrieve` → RAG subgraph, keyed on the question), *compute
the tax* (`calculate`, keyed on the `tax_profile` `plan` extracts), *compose a
cited answer* (`synthesize`). `triage` selects which subset a question needs. On
`rag_plus_calc` the two subtasks share no inputs, so `plan`'s conditional edge
returns `["retrieve", "calculate"]` — LangGraph dispatches both in one superstep
and they execute independently, then `synthesize` runs once both have landed
(`route_after_plan` in `app/graph/nodes/plan.py`; `test_graph_end_to_end.py`
asserts the fan-out). The wall-clock win is marginal here — `calculate` is a
sub-millisecond deterministic call — so the value is architectural: a real
parallel branch with a fan-in join. The RAG subgraph decomposes once more, one
level down, expanding the question into several targeted sub-queries.

- **triage** — LLM classifies the question into `rag_only` / `needs_calc` /
  `rag_plus_calc` / `out_of_scope` (autonomous routing).
- **plan** — extracts a structured `tax_profile` (filing status, income,
  dependents, tax year) — the input the `calculate` subtask needs. Skipped on
  `rag_only` / `out_of_scope`.
- **retrieve** — invokes the modular **RAG subgraph**. Tool #1 (retrieval). Runs
  on `rag_only` and (in parallel) `rag_plus_calc`.
- **calculate** — deterministic `estimate_tax(...)`. Tool #2 (non-retrieval).
  Runs on `needs_calc` and (in parallel) `rag_plus_calc`; a no-op if `plan`
  produced no `tax_profile`.
- **synthesize** — LLM composes a cited answer from context and/or calc result.
- **guardrails** — deterministic, no model call: redacts SSNs, and flags
  citations to pages that were never retrieved and dollar amounts that trace to
  neither the calculator nor the retrieved context.
- **validate** — checks citations, numeric consistency, and the guardrail
  findings; on failure it sends exactly one retry back to `retrieve`
  (re-running retrieve -> synthesize -> guardrails; `calculate` does not re-run,
  its result persists in state), then ends regardless of the second result.

Out of scope for this prototype (production would add them, likely as a
dedicated pre/post model): input moderation, prompt-injection / jailbreak
screening, an LLM-judge faithfulness check, rate limiting, and an abuse policy.

### RAG subgraph (separate compiled graph, `app/rag/subgraph.py`)

```
expand_query --> retrieve_candidates --> rerank --> grade_docs --(loop: broaden & retry | continue)--> assemble_context
```

Compiled node set (verified): `expand_query, retrieve_candidates, rerank,
grade_docs, assemble_context` (plus `__start__` / `__end__`). Each node is one
named RAG subsystem: query expansion (condenses a follow-up against chat history
into a standalone question, then LLM rewrite + deterministic domain hints),
candidate retrieval (dense + BM25 + RRF fusion), cross-encoder reranking (with
a table-first boost for amount questions), relevance grading, and context
assembly with citations. The `candidates → reranked → kept` counts flow back to
the API (`retrieval_funnel`) and show in the Streamlit trace panel.

### Tools

| Tool | File | Kind |
|------|------|------|
| Retriever (RAG subgraph) | `app/tools/retriever_tool.py` | retrieval |
| Federal tax estimator | `app/tools/tax_calculator.py` | non-retrieval, deterministic |

### Design rationale (highlights)

- **Approach A (fixed agentic graph) over a ReAct agent** — small local models
  tool-call unreliably; an explicit graph makes routing and retries auditable.
  Tools are invoked structurally by nodes rather than through LLM tool-calls for
  the same reason.
- **Local `sentence-transformers` embeddings + embedded Chroma** — no API key,
  deterministic, metadata-aware, runs in the same process as the API.
- **Structure-aware chunking** — tax rules are hierarchical and citations need
  `section` + `page`, so chunks carry publication / section / page metadata.
- **Pluggable `ollama` / `dummy` LLM** — reconciles "real local model" with
  "reproducible and offline for graders". `dummy` is the default for tests,
  eval, and load tests.
- **`llama3.2:3b` as the local model** — trade-off: it runs on ~4 GB RAM with no
  GPU (fits a laptop / CI box), at the cost of ~3–4 min/answer on CPU and shakier
  classification — the routing guard in `triage` exists to backstop that. An 8B
  (llama3.1, qwen2.5) routes and writes better but needs more RAM and roughly
  doubles latency on CPU; not worth it for a prototype whose answers are already
  gated by a deterministic calculator and citation checks. Swap via `LLM_MODEL`.
- **Deterministic calculator, never LLM arithmetic** — tax math must be exact
  and testable.

## Evaluation and performance — results

Both artifacts below are generated, not hand-written:
`LLM_MODE=dummy python -m eval.run_eval` and
`LLM_MODE=dummy python -m loadtest.run_load`.

### Functional evaluation — `docs/eval-results.md` (+ `.json`)

15 questions covering the three routes that run end to end (`rag_only`,
`rag_plus_calc`, `out_of_scope`). `run_eval` writes both a markdown table and a
machine-readable `docs/eval-results.json`. Numbers below are the committed run
under `LLM_MODE=ollama` / `llama3.2:3b`.

| Metric | Result | Note |
|--------|--------|------|
| Route accuracy | 100% (15/15) | |
| Retrieval hit rate (cited pub matches expected) | 100% | |
| Citation rate | 100% | |
| Numeric accuracy (calc questions) | 100% (3/3) | |
| Keyword hit rate | 60.0% | brittle substring proxy vs. 3B phrasing; low signal |

Route accuracy and `tax_profile` extraction depend on the LLM, so this runs
under `LLM_MODE=ollama`; retrieval hit rate and citation rate are largely
independent of generation quality. Keyword hit rate is a weak `all(k in answer)`
substring check kept only as a smoke signal — it is not a headline number.

**`needs_calc` (pure calculation, no retrieval)** is validated at the node level
in `tests/test_graph_triage.py` rather than in this end-to-end set. The spec
allows evaluating "a single node or the entire workflow", and with a real 3B the
label is unstable on calc questions that resemble the `rag_plus_calc` examples;
the triage guard deliberately biases every dollar-amount question toward
`rag_plus_calc` so the returned figure always carries a citation. The calculator
path itself is covered by `tests/test_tax_calculator.py` and by q08–q10.

### Retrieval quality — `## Retrieval quality` section of the same file

Judged at `(publication, page)` granularity against the `relevant_pages` labels
in `eval/questions.yaml`, by invoking the RAG subgraph directly and reading its
`raw_hits` (post RRF fusion) → `reranked_hits` (post cross-encoder) →
`graded_hits` (kept for the prompt).

| Metric | Meaning | Committed run (k=5, 14 labelled Qs) |
|--------|---------|------|
| `precision_at_k` / `recall_at_k` | of the reranked top-k (k = `search_k`) | 0.31 / 0.61 |
| `mrr` | mean reciprocal rank of the first relevant page | 0.59 |
| `context_precision` | fraction of the assembled context that is relevant | 0.31 |
| `mrr_fused` → `mrr_reranked` (`rerank_mrr_lift`) | ranking gain the `rerank` node adds over pure RRF | 0.45 → 0.59 (**+0.13**) |
| `hit_rate_fused` → `hit_rate_reranked` | same, as a top-k hit rate | 0.79 → 0.79 (rerank reorders within top-5, doesn't add new pages) |

Retrieval reliably finds the right *publication* (hit rate 100% in the
functional eval) but page-level precision is mediocre — the weak spots are the
table-heavy pages (standard-deduction table, rate schedules). Improving page
recall (larger rerank pool, ±1-page label tolerance, better table chunking) is
the main retrieval lever; the cross-encoder rerank already contributes a clear
+0.13 MRR.

### Load test — `docs/loadtest-results.md`

100 requests, `LLM_MODE=dummy`, **concurrency 4**, 0 errors.

| p50 | p90 | p95 | p99 | mean | max | throughput |
|-----|-----|-----|-----|------|-----|------------|
| 19 ms | 656 ms | 1066 ms | 10450 ms | 528 ms | 10562 ms | 7.6 req/s |

Steady-state p50 is ~19 ms; the p99/max tail is the first few requests all
queuing behind the one-time model + HNSW warmup, since retrieval is serialised
by a single lock (below). A latency histogram is written to
`docs/loadtest-latency.png` locally (gitignored).

**Bottleneck node: `retrieve`** — mean ~695 ms per request, essentially the
entire request budget (`triage`, `plan`, `calculate`, `synthesize`,
`guardrails`, `validate` are ~0 ms in `dummy` mode). In `dummy` mode this cost
is vector search over the embedded Chroma index; in `ollama` mode `synthesize`
(LLM generation on CPU) typically dominates instead.

**Caching** (`app/rag/cache.py`, in-process, `CACHE_ENABLED=false` to disable):

- **query-embedding LRU** — skips re-encoding a query string already seen
  (expansion + domain hints make queries repeat).
- **RAG-subgraph result LRU** — a repeated question skips the whole subgraph
  (expansion LLM call + dense + BM25 + RRF + rerank).

Both keys carry an `index_fingerprint()` (retrieval config + live chunk count),
so a re-ingest or a knob change invalidates them automatically. Hit/miss counts
are on `/metrics` as `rag_cache_events_total`.

Further optimizations not done here: an end-to-end response cache keyed on
normalized question + route (skips synthesis too), and replacing the
`grade_docs` step with a pure score threshold. Moving both caches to Redis is
the step that lets multiple API workers share them (see concurrency below).

**Concurrency.** The API is one `uvicorn` process; FastAPI runs the sync
`/chat` handler on its threadpool, so concurrent requests do run on separate
threads — but they share process-global state that isn't built for that: one
`chromadb.PersistentClient` (SQLite + hnswlib), one `SentenceTransformer`, one
`CrossEncoder`, the BM25 index, the `_compiled` graph. Left unguarded, the
retrieval path segfaulted the worker at concurrency ≥ 2 (native crash, no
traceback) — the shared Chroma client the prime suspect.

*Implemented — #1, serialise the unsafe section.* One process-wide
`threading.RLock` (`app/rag/retriever.py`) around every Chroma call and every
torch forward pass (`.encode` / `.predict`). Retrieval is now serial (~700 ms
in the load test, dominated by the cold model/HNSW warmup the first requests
queue behind); the minutes-long LLM synthesis still overlaps across threads.
BM25 and the loaded models are read-only and safe to share — only the Chroma
client and inference are guarded. Load test now runs clean at concurrency 4.

*Design — not implemented (infrastructure, not RAG):*

2. *Per-worker Chroma clients* — one `PersistentClient` per thread/worker on
   the same read-only path (SQLite is multi-reader), so retrieval parallelises;
   a small model pool or a lock kept only around inference.
3. *Async request path* — `async def chat` + `await _compiled.ainvoke` /
   `astream` (LangGraph supports both; sync nodes run in a threadpool). LLM
   calls move to `ollama.AsyncClient` — the real win, since a request awaiting
   a multi-minute generation no longer holds a thread. CPU-bound bits go
   through `asyncio.to_thread` / a `ProcessPoolExecutor`; Chroma stays wrapped
   with #2.
4. *Scale out* — retrieval + models as a separate service (HTTP/gRPC) so the
   agent API holds no native state and runs `uvicorn --workers N` behind a
   proxy; the LRUs move to **Redis**. Ollama then becomes the ceiling — multiple
   replicas or a batching server (vLLM/TGI), out of scope for a local
   no-paid-API prototype.

Realistic path: #1 is done; #3 + #2 are the target; #4 only for real traffic.

**No request cancellation.** `/chat` runs the graph to completion regardless of
the client; there is no abort path (it would need the async/multi-worker rework
above, plus a cancellation flag checked between nodes or an aborted Ollama
call). `/chat/stream` mitigates the *experience* — the Streamlit UI shows each
node completing live via SSE instead of one opaque spinner — but the work still
finishes server-side.

## Install and run

Python 3.11. No paid APIs. `LLM_MODE=dummy` is the default for offline,
reproducible runs; `LLM_MODE=ollama` needs the model pulled first.

### Prerequisites

Docker + Docker Compose. ~4 GB free RAM for the 3B model (not needed in
`dummy` mode).

### First run — reproducible / offline (recommended)

```bash
cp .env.example .env
make ingest                                            # build the vector index (one-time, needs internet)
LLM_MODE=dummy make up                                 # UI :8501, API :8000 — deterministic stub LLM
```

`make ingest` downloads IRS Pub. 17 / 501 / 505 and builds the embedded Chroma
index under `data/chroma/`. It needs internet once; the SHA-256 of each PDF is
pinned in `app/ingest/sources.yaml`. After that the stack runs fully offline,
and `LLM_MODE=dummy` makes every answer deterministic — this is the path to
reproduce the test suite and the retrieval eval.

### With the real model (for actual answers)

```bash
docker compose exec ollama ollama pull llama3.2:3b     # ~2 GB, one-time
make up                                                # LLM_MODE defaults to ollama
```

Answers then depend on `llama3.2:3b` (non-deterministic, ~3–4 min/answer on CPU).

### Local (no Docker)

```bash
make install
python -m app.ingest.build_index                       # needs internet, one time
LLM_MODE=dummy uvicorn app.api.main:app --port 8000
LLM_MODE=dummy streamlit run app/ui/streamlit_app.py   # separate shell
```

### Observability (optional)

The core stack (`make up` / `docker compose up`) is just `ollama`, `api`, `ui`.
The monitoring stack is a separate compose profile — nothing in the app path
depends on it:

```bash
make up-obs      # docker compose --profile observability up --build
# Grafana http://localhost:3001 (anon)  |  Prometheus :9090  |  Loki :3100  |  Pushgateway :9091
```

The API exposes Prometheus metrics at `/metrics`; the provisioned Grafana
dashboard ("Agentic RAG Tax Chatbot") shows:

| Group | Metrics |
|-------|---------|
| Request | `rag_request_duration_seconds` (p95 by route), `rag_requests_total` (throughput, error ratio) |
| Graph | `rag_node_duration_seconds` (7 main nodes), `rag_subgraph_node_duration_seconds` (5 RAG nodes), `rag_validate_retries_total` |
| LLM | `rag_llm_duration_seconds` (p95 by op), `rag_llm_calls_total` (by op), `rag_llm_tokens` (prompt/output), `rag_llm_fallback_total` (Ollama-unreachable → dummy) |
| Retrieval | `rag_context_words`, `rag_retrieval_chunks` |
| Offline eval | `rag_eval_*` gauges — `run_eval` pushes route accuracy, retrieval hit rate, Precision@k, Recall@k, MRR and rerank lift to the pushgateway when `PROM_PUSHGATEWAY` is set |

```bash
PROM_PUSHGATEWAY=localhost:9091 OLLAMA_BASE_URL=http://localhost:11434 \
  LLM_MODE=ollama python -m eval.run_eval      # results also land in Grafana
```

### Structured logging

`app/observability/logging.py` emits one JSON line per pipeline stage on stdout,
all sharing a `request_id`, so a single `/chat` call is greppable end to end:
`api.request.received` → `triage.routed` (llm route vs. final) → `rag.cache` →
`expand_query.expanded` (the rewritten queries) → `retrieve_candidates.fused`
(pool size + top `(chunk_id, pub, page, score)`) → `rerank.reranked` (scores
before/after) → `rag.context` (citations, context words) → per-node `*.done`
with timings → `api.request.completed`. `LOG_LEVEL=DEBUG` adds the heavy
payloads; SSNs are redacted unless `LOG_PII=true`; `LOG_JSON=false` for plain
text in local dev.

In the observability profile, **Promtail** ships every container's stdout to
**Loki**, wired into Grafana as a datasource. Trace one request in Grafana →
Explore → Loki:

```logql
{service="api"} | json | request_id = `a1b2c3d4`
```

`container`, `service`, `level` and `stage` are indexed labels; `request_id`,
`question`, `queries`, `top`, … are parsed from the JSON at query time.

### Tests / eval / load test

```bash
make test                                              # LLM_MODE=dummy python -m pytest -q
LLM_MODE=dummy python -m eval.run_eval                  # writes docs/eval-results.md
LLM_MODE=dummy python -m loadtest.run_load --api-url http://localhost:8000 --n 100 --concurrency 4
# API must be running. Retrieval is lock-serialised (see "Concurrency"), so higher
# concurrency mostly lengthens the warmup queue rather than the steady state.
```

## Further extensions

Deliberately left out to keep the prototype lean; each is a bounded add-on:

- **Concurrency** — #1 (lock the Chroma/model critical section) is done; the
  async request path + per-worker clients, and a retrieval service +
  multi-worker + Redis to scale, are designed under *Concurrency* above.
- **Request cancellation** — a cancel endpoint that trips a flag checked between
  nodes / aborts the Ollama call; depends on the async rework above.
- **Langfuse** (LLM trace tree) — `run_agent` / `run_agent_stream` already take a
  `callbacks` list threaded into `_compiled.invoke(config={"callbacks": …})`.
  Add `langfuse`, wire a `CallbackHandler`, and run its Postgres-backed stack in
  its own compose profile. Dropped here because Loki (`| json | request_id=…`)
  plus the step panel already cover request tracing, and it needs Postgres.
- **Redis** for a shared response/retrieval cache across workers (the in-process
  LRUs become a fallback).
- **Retrieval** — larger rerank pool + `R@10`, ±1-page label tolerance, better
  table/heading chunking to lift page-level recall.
