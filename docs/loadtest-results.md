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

## Notes

- Retrieval is serialised by one process-wide lock (see README "concurrency"), so at concurrency > 1 the first requests queue behind the one-time model + HNSW warmup — that is the p99/max tail here. Steady state (p50) is unaffected.
- Not yet done: an end-to-end response cache keyed on normalised question + route (skips synthesis too), and skipping the `validate` retry when the draft already carries citations and the calc number.
