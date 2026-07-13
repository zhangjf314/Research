# Retrieval Ablation v1

- Dataset: `gold-set-v1-human-reviewed-2026-07-13`
- Approved: 50/50
- Evaluated answerable queries: 48
- Approved unanswerable records excluded from retrieval metrics: 2
- Reranker: disabled
- LLM calls: none

| Variant | Hit@1 | Hit@5 | Recall@5 | Recall@10 | MRR | NDCG@10 | Mean ms | P50 ms | P95 ms | Failures | Points |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hash_structural_dense | 0.062 | 0.167 | 0.167 | 0.292 | 0.112 | 0.153 | 13.336 | 3.702 | 47.556 | 0 | 2065 |
| hash_structural_hybrid | 0.104 | 0.208 | 0.208 | 0.302 | 0.158 | 0.191 | 23.645 | 12.726 | 56.249 | 0 | 2065 |
| jina_structural_dense | 0.042 | 0.146 | 0.135 | 0.208 | 0.082 | 0.111 | 461.735 | 418.289 | 568.862 | 0 | 2064 |
| jina_structural_hybrid | 0.021 | 0.125 | 0.125 | 0.292 | 0.082 | 0.131 | 436.747 | 420.941 | 509.930 | 0 | 2064 |

## Measured deltas

| Retriever | Metric | Jina - Hash |
|---|---|---:|
| dense | hit_at_1 | -0.021 |
| dense | hit_at_5 | -0.021 |
| dense | recall_at_5 | -0.031 |
| dense | recall_at_10 | -0.083 |
| dense | mrr | -0.031 |
| dense | ndcg_at_10 | -0.042 |
| hybrid | hit_at_1 | -0.083 |
| hybrid | hit_at_5 | -0.083 |
| hybrid | recall_at_5 | -0.083 |
| hybrid | recall_at_10 | -0.010 |
| hybrid | mrr | -0.075 |
| hybrid | ndcg_at_10 | -0.060 |

## Query examples

### Semantic improvements

- `q013`: Hash rank >10, Jina rank 2 — What method or technical approach does the target paper propose?
- `q042`: Hash rank 10, Jina rank 2 — What are the target paper's main contributions?
- `q044`: Hash rank >10, Jina rank 3 — How are the target paper's experiments designed and evaluated?

### Regressions

- `q037`: Hash rank 1, Jina rank >10 — What are the target paper's main contributions?
- `q024`: Hash rank 2, Jina rank >10 — How are the target paper's experiments designed and evaluated?
- `q021`: Hash rank 1, Jina rank 8 — What research problem does the target paper address?

## Failure examples

- `q001`: neither hybrid variant retrieved a gold paper in Top-10 — What research problem does the target paper address?
- `q002`: neither hybrid variant retrieved a gold paper in Top-10 — What are the target paper's main contributions?
- `q003`: neither hybrid variant retrieved a gold paper in Top-10 — What method or technical approach does the target paper propose?
- `q004`: neither hybrid variant retrieved a gold paper in Top-10 — How are the target paper's experiments designed and evaluated?
- `q006`: neither hybrid variant retrieved a gold paper in Top-10 — What research problem does the target paper address?

## Evaluation validity note

Many gold questions deliberately use generic phrases such as `the target paper` and do not include a title, arXiv ID, or topic-bearing subject. In an unrestricted 36-paper corpus retrieval task, those queries do not identify which paper is intended. The scores are reproducible for this dataset, but they must not be interpreted as a general model leaderboard until the protocol either supplies a non-gold paper scope filter or rewrites questions to be independently identifiable.

## Stage 11B recommendation

Do not promote Jina Hybrid as the Stage 11B default from this run; retain Hash Hybrid until the measured MRR, NDCG@10, and recall acceptance condition is met.
This decision is generated only from the measured hybrid metrics; no reranker or LLM result is involved.
