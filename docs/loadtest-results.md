# Load Test Results

- Requests: 100  |  Concurrency: 1  |  Errors: 0
- Throughput: 9.89 req/s

## Latency (ms)

| p50 | p90 | p95 | p99 | mean | max |
|-----|-----|-----|-----|------|-----|
| 37.4 | 67.3 | 72.1 | 150.2 | 101.0 | 6700.6 |

## Per-node mean (ms)

- retrieve: 108.8
- triage: 0.0
- calculate: 0.0
- synthesize: 0.0
- validate: 0.0
- plan: 0.0

## Bottleneck

`retrieve` is the dominant per-request cost (mean 108.8 ms). In `dummy` mode this is vector search over the embedded Chroma index; in `ollama` mode LLM generation in `synthesize` typically dominates instead.

## Optimization recommendations

1. Cut LLM calls on the retrieval path: replace the `grade_docs` LLM grader with the score threshold only, and merge `expand_query` into a single call — removes ~2 LLM round-trips per request.
2. Add a semantic response cache keyed on normalized question + route; skip the `validate` retry when citations are present and the calc number is already in the draft.
