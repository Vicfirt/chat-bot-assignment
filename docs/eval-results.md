# Functional Evaluation Results

Items: 15

| id | route_ok | retrieval_hit | has_citation | number_ok | keyword_hit |
|----|----------|---------------|--------------|-----------|-------------|
| q01 | True | True | True | None | True |
| q02 | True | True | True | None | False |
| q03 | True | True | True | None | True |
| q04 | True | True | True | None | True |
| q05 | True | True | True | None | False |
| q06 | True | True | True | None | True |
| q07 | True | True | True | None | True |
| q08 | True | True | True | True | False |
| q09 | True | True | True | True | False |
| q10 | True | True | True | True | False |
| q11 | True | True | True | None | False |
| q12 | True | True | True | None | True |
| q13 | True | True | True | None | True |
| q14 | True | True | True | None | True |
| q15 | True | True | True | None | True |

## Aggregate

- **n**: 15
- **route_accuracy**: 100.0
- **retrieval_hit_rate**: 100.0
- **citation_rate**: 100.0
- **numeric_accuracy**: 100.0
- **keyword_hit_rate**: 60.0

## Retrieval quality (RAG subgraph, k=5, 14 page-labeled questions)

| id | P@k | R@k | MRR | context_precision | RR fused | RR reranked |
|----|-----|-----|-----|-------------------|----------|-------------|
| q01 | 0.400 | 1.000 | 1.000 | 0.400 | 0.333 | 1.000 |
| q02 | 0.400 | 1.000 | 1.000 | 0.400 | 0.333 | 1.000 |
| q03 | 0.400 | 1.000 | 0.500 | 0.400 | 0.500 | 0.500 |
| q04 | 0.000 | 0.000 | 0.000 | 0.000 | 0.200 | 0.000 |
| q05 | 0.200 | 0.500 | 0.200 | 0.200 | 0.167 | 0.200 |
| q06 | 0.250 | 1.000 | 1.000 | 0.250 | 1.000 | 1.000 |
| q07 | 0.200 | 0.500 | 0.250 | 0.200 | 0.200 | 0.250 |
| q08 | 0.000 | 0.000 | 0.000 | 0.000 | 0.059 | 0.000 |
| q09 | 0.400 | 0.500 | 1.000 | 0.400 | 0.200 | 1.000 |
| q10 | 0.000 | 0.000 | 0.000 | 0.000 | 0.020 | 0.000 |
| q11 | 0.250 | 1.000 | 1.000 | 0.250 | 1.000 | 1.000 |
| q12 | 0.600 | 0.500 | 1.000 | 0.600 | 1.000 | 1.000 |
| q13 | 0.200 | 0.500 | 0.250 | 0.200 | 0.333 | 0.250 |
| q14 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### Aggregate

- **k**: 5
- **n_labeled**: 14
- **precision_at_k**: 0.307
- **recall_at_k**: 0.607
- **mrr**: 0.586
- **context_precision**: 0.307
- **mrr_fused**: 0.453
- **mrr_reranked**: 0.586
- **hit_rate_fused**: 0.786
- **hit_rate_reranked**: 0.786
- **rerank_mrr_lift**: 0.133
- **rerank_hit_lift**: 0.0
