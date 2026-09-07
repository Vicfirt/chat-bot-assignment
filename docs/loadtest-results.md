# Load Test Results

- Requests: 100  |  Concurrency: 1  |  Errors: 0
- Throughput: 11.33 req/s

## Latency (ms)

| p50 | p90 | p95 | p99 | mean | max |
|-----|-----|-----|-----|------|-----|
| 37.6 | 68.5 | 74.1 | 147.0 | 88.2 | 5365.8 |

## Per-node mean (ms)

- retrieve: 94.6
- triage: 0.0
- calculate: 0.0
- synthesize: 0.0
- validate: 0.0
- plan: 0.0

## Bottleneck

`retrieve` dominates per-request time (mean 94.6 ms). This is LLM generation on CPU via Ollama.

## Optimization recommendations

1. Cut LLM calls on the retrieval path: replace the `grade_docs` LLM grader with the score threshold only, and merge `expand_query` into a single call — removes ~2 LLM round-trips per request.
2. Add a semantic response cache keyed on normalized question + route; skip the `validate` retry when citations are present and the calc number is already in the draft.
