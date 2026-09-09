# Load Test Results

- `LLM_MODE=dummy`  |  Requests: 100  |  Concurrency: 4  |  Errors: 0  |  Warmup discarded: 5
- Throughput: 5.15 req/s

## Latency (ms) — first 5 requests excluded

| p50 | p90 | p95 | p99 | mean | max |
|-----|-----|-----|-----|------|-----|
| 995.9 | 1269.9 | 1336.3 | 1657.1 | 751.7 | 1661.7 |

## Per-node mean (ms)

- retrieve: 1072.1
- triage: 0.0
- plan: 0.0
- calculate: 0.0
- synthesize: 0.0
- guardrails: 0.0
- validate: 0.0

## Bottleneck

`retrieve` dominates at 1072 ms mean; every other node is ~0 ms.

In `LLM_MODE=ollama` the request is LLM generation on CPU. `synthesize` (the ~400-token cited answer) is the largest single node; `retrieve` here is 1072 ms because it includes the RAG subgraph's own `expand_query` LLM rewrite, and `triage` is a third LLM call. Their relative weight shifts with model size — a small model makes them a few seconds each, a larger one makes `expand_query` alone minutes. The deterministic nodes (`plan`, `calculate`, `guardrails`, `validate`) are ~0 ms. In `LLM_MODE=dummy` the LLM nodes collapse to ~0 ms and `retrieve` (vector search + rerank, lock-serialised) is all that is left — that run isolates the retrieval path.

## Optimization recommendations

1. **Attack `synthesize`.** (a) An **end-to-end response cache** keyed on the normalised question, checked before the graph — *implemented* in `app/api/main.py`: a repeat `/chat` returns in ~1 ms instead of tens of seconds to minutes. (b) **Token streaming** from `synthesize` — the `/chat/stream` SSE plumbing already carries step events; extending it to model tokens would drop time-to-first-token to ~1-2 s. Not done.
2. **Make `triage` and `expand_query` rules/embeddings-only.** They are two more LLM round-trips (100% of this run). The deterministic guard in `triage` already overrides the model for the common cases, and `expand_query`'s rewrite buys little over the domain-hint + `tax_profile` queries. Bonus: routing becomes deterministic. This is the largest saving on a bigger model, where `expand_query` alone runs into minutes.

## Note on the tail

- Retrieval is serialised by one process-wide lock (see README "concurrency"). Under `ollama` with concurrency > 1 the single CPU model is the real serialisation point — queued requests can exceed the client timeout, which is why the real-model profile is run at concurrency 1. Under `dummy` the tail is the one-time model + HNSW warm-up the first requests queue behind; steady-state p50 is unaffected.
