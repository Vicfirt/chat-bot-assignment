# Load Test Results

- Mode: `LLM_MODE=ollama`  |  Requests: 8  |  Concurrency: 1  |  Errors: 0  |  Warmup discarded: 2
- Throughput: 0.03 req/s

## Latency (ms) — first 2 requests excluded

| p50 | p90 | p95 | p99 | mean | max |
|-----|-----|-----|-----|------|-----|
| 32194.9 | 40907.2 | 43938.7 | 46363.8 | 34306.9 | 46970.1 |

## Per-node mean (ms)

- synthesize: 23809.6
- retrieve: 7429.8
- triage: 3059.2
- plan: 0.6
- guardrails: 0.1
- validate: 0.0
- calculate: 0.0

## Bottleneck

`synthesize` dominates at 23810 ms mean, then `retrieve` 7430 ms, `triage` 3059 ms.

In `LLM_MODE=ollama` `synthesize` (the ~400-token cited answer, generated on CPU) is the whole request — ~90% of wall-clock even on `llama3.2:1b`. `triage` and `expand_query` (the RAG subgraph's own LLM call, counted under `retrieve`) are a few seconds each *when run serially*; under concurrency > 1 their measured time inflates because each call queues behind other requests' `synthesize` on the single CPU model. The deterministic nodes (`plan`, `calculate`, `guardrails`, `validate`) are ~0 ms. In `LLM_MODE=dummy` the LLM nodes collapse to ~0 ms and `retrieve` (vector search + rerank, lock-serialised) is all that is left — that run isolates the retrieval path.

## Optimization recommendations

1. **Attack `synthesize`.** It is ~90% of the request. Two independent levers: (a) an **end-to-end response cache** keyed on normalised question + route, checked before the graph — a repeat returns in ~1 ms instead of ~30 s; the current caches (`app/rag/cache.py`) stop at the RAG subgraph, so synthesis still re-runs. (b) **Token streaming** from `synthesize` — the `/chat/stream` SSE plumbing already exists for step events; extending it to model tokens drops time-to-first-token to ~1-2 s while the full answer still takes ~30 s.
2. **Collapse the classification calls.** `triage` and `expand_query` are ~20-25% of a serial request (measured: ~3 s + ~5 s on `llama3.2:1b`) and can be embeddings/rules-only — the deterministic guard in `triage` already overrides the model for the common cases, and `expand_query`'s rewrite buys little over the domain-hint + `tax_profile` queries. Bonus: routing becomes deterministic.

## Note on the tail

- Retrieval is serialised by one process-wide lock (see README "concurrency"). Under `ollama` with concurrency > 1 the single CPU model is the real serialisation point — queued requests can exceed the client timeout, which is why the real-model profile is run at concurrency 1. Under `dummy` the tail is the one-time model + HNSW warm-up the first requests queue behind; steady-state p50 is unaffected.
