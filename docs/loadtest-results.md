# Load Test Results

- Requests: 100  |  Concurrency: 4  |  Errors: 0
- Throughput: 7.57 req/s

## Latency (ms)

| p50 | p90 | p95 | p99 | mean | max |
|-----|-----|-----|-----|------|-----|
| 19.2 | 656.3 | 1066.1 | 10450.2 | 528.4 | 10561.6 |

## Per-node mean (ms)

- retrieve: 694.7
- triage: 0.0
- calculate: 0.0
- synthesize: 0.0
- guardrails: 0.0
- validate: 0.0
- plan: 0.0

## Bottleneck

`retrieve` is the dominant per-request cost (mean 694.7 ms). In `dummy` mode this is vector search over the embedded Chroma index; in `ollama` mode LLM generation in `synthesize` typically dominates instead.

## Optimization recommendations

1. **End-to-end response cache** keyed on normalised question + route, returning the stored answer before the graph runs. The current caches (`app/rag/cache.py`) stop at the RAG subgraph; synthesis, guardrails and validate still execute on a repeat. A full-response cache takes a repeat query to ~1 ms and, in `ollama` mode, removes the dominant ~3-4 min `synthesize` cost entirely on the hit path.
2. **Skip the `validate` retry** when the draft already carries the expected citations and the calc total. The retry re-runs retrieve -> calculate -> synthesize -> guardrails — a second full LLM round-trip — and rarely changes a draft that already passed the cheap checks. Gating it on "cheap checks already green" removes that tail for the common case.

## Note on the tail

- Retrieval is serialised by one process-wide lock (see README "concurrency"), so at concurrency > 1 the first requests queue behind the one-time model + HNSW warmup — that is the p99/max tail here. Steady state (p50) is unaffected.
