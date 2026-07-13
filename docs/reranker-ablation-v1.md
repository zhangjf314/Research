# Reranker Ablation v1

- Fixed retrieval: Jina Embedding + Structural Hybrid
- Shared initial candidates: Top-30
- Rerank output retained in Trace: Top-30
- Evaluation cutoff: Top-10
- Corpus: production-corpus-v1, 34 documents, 2062 points
- Protocol: retrieval-gold-v2, pending review 0
- Formal fallback: disabled
- LLM calls: none
- Deep Research: not run

## Paper-scoped metrics

| Variant | Hit@1 | Hit@5 | Recall@5 | Recall@10 | MRR | NDCG@10 | Pre Recall@10 | Post Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_rerank | 0.152 | 0.391 | 0.232 | 0.307 | 0.245 | 0.158 | 0.307 | 0.307 |
| lexical_rerank | 0.130 | 0.348 | 0.154 | 0.309 | 0.231 | 0.154 | 0.307 | 0.309 |
| jina_reranker_v3 | 0.152 | 0.348 | 0.214 | 0.318 | 0.243 | 0.155 | 0.307 | 0.318 |

## Multi-paper metrics

| Variant | Coverage@5 | All@5 | Evidence Recall@5 | Coverage@10 | All@10 | Evidence Recall@10 |
|---|---:|---:|---:|---:|---:|---:|
| no_rerank | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 0.600 |
| lexical_rerank | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 0.600 |
| jina_reranker_v3 | 1.000 | 1.000 | 0.200 | 1.000 | 1.000 | 0.700 |

## Latency and reliability

| Variant | Retrieval mean/p50/p95 ms | Rerank mean/p50/p95 ms | Total mean/p50/p95 ms | Failures | Fallbacks | API requests |
|---|---:|---:|---:|---:|---:|---:|
| no_rerank | 463.5/403.0/699.3 | 0.0/0.0/0.0 | 463.5/403.0/699.3 | 0 | 0 | 0 |
| lexical_rerank | 463.5/403.0/699.3 | 4.6/4.7/5.3 | 468.2/408.2/704.6 | 0 | 0 | 0 |
| jina_reranker_v3 | 463.5/403.0/699.3 | 7034.3/717.2/63236.6 | 7497.8/1144.5/63661.6 | 0 | 0 | 55 |

## Category breakdown

| Variant | Category | Answerable | Hit@5 | Recall@10 | MRR |
|---|---|---:|---:|---:|---:|
| no_rerank | algorithm_steps | 5 | 0.600 | 0.500 | 0.333 |
| no_rerank | experiment_results | 4 | 0.000 | 0.042 | 0.036 |
| no_rerank | experiment_setup | 5 | 0.800 | 0.607 | 0.398 |
| no_rerank | limitations | 7 | 0.714 | 0.571 | 0.550 |
| no_rerank | method | 5 | 0.400 | 0.180 | 0.167 |
| no_rerank | multi_paper_comparison | 2 | 1.000 | 0.600 | 0.667 |
| no_rerank | paper_contributions | 10 | 0.100 | 0.100 | 0.117 |
| no_rerank | research_background | 10 | 0.300 | 0.250 | 0.161 |
| no_rerank | unanswerable | 0 | N/A | N/A | N/A |
| lexical_rerank | algorithm_steps | 5 | 0.600 | 0.450 | 0.250 |
| lexical_rerank | experiment_results | 4 | 0.000 | 0.062 | 0.042 |
| lexical_rerank | experiment_setup | 5 | 0.400 | 0.567 | 0.217 |
| lexical_rerank | limitations | 7 | 0.571 | 0.548 | 0.488 |
| lexical_rerank | method | 5 | 0.400 | 0.180 | 0.300 |
| lexical_rerank | multi_paper_comparison | 2 | 1.000 | 0.600 | 0.200 |
| lexical_rerank | paper_contributions | 10 | 0.200 | 0.133 | 0.137 |
| lexical_rerank | research_background | 10 | 0.300 | 0.283 | 0.186 |
| lexical_rerank | unanswerable | 0 | N/A | N/A | N/A |
| jina_reranker_v3 | algorithm_steps | 5 | 0.400 | 0.517 | 0.445 |
| jina_reranker_v3 | experiment_results | 4 | 0.000 | 0.042 | 0.031 |
| jina_reranker_v3 | experiment_setup | 5 | 0.800 | 0.740 | 0.429 |
| jina_reranker_v3 | limitations | 7 | 0.571 | 0.643 | 0.235 |
| jina_reranker_v3 | method | 5 | 0.400 | 0.200 | 0.267 |
| jina_reranker_v3 | multi_paper_comparison | 2 | 1.000 | 0.700 | 0.667 |
| jina_reranker_v3 | paper_contributions | 10 | 0.200 | 0.133 | 0.211 |
| jina_reranker_v3 | research_background | 10 | 0.200 | 0.133 | 0.160 |
| jina_reranker_v3 | unanswerable | 0 | N/A | N/A | N/A |

## Difficulty breakdown

| Variant | Difficulty | Answerable | Hit@5 | Recall@10 | MRR |
|---|---|---:|---:|---:|---:|
| no_rerank | easy | 18 | 0.278 | 0.164 | 0.150 |
| no_rerank | hard | 24 | 0.542 | 0.445 | 0.360 |
| no_rerank | medium | 6 | 0.333 | 0.278 | 0.208 |
| lexical_rerank | easy | 18 | 0.333 | 0.213 | 0.184 |
| lexical_rerank | hard | 24 | 0.458 | 0.442 | 0.281 |
| lexical_rerank | medium | 6 | 0.167 | 0.167 | 0.167 |
| jina_reranker_v3 | easy | 18 | 0.333 | 0.219 | 0.297 |
| jina_reranker_v3 | hard | 24 | 0.500 | 0.503 | 0.299 |
| jina_reranker_v3 | medium | 6 | 0.000 | 0.000 | 0.000 |

## Jina ranking changes

- Improved queries: 11
- Regressed queries: 14
- Unchanged queries: 23

### Improvements

- `q038`: >10 → 1 — What method or technical approach does the target paper propose?
- `q041`: >10 → 2 — What research problem does the target paper address?
- `q043`: >10 → 3 — What method or technical approach does the target paper propose?
- `q027`: 6 → 1 — What are the target paper's main contributions?
- `q014`: 5 → 1 — How are the target paper's experiments designed and evaluated?

### Regressions

- `q006`: 1 → >10 — What research problem does the target paper address?
- `q028`: 1 → 10 — What method or technical approach does the target paper propose?
- `q004`: 1 → 9 — How are the target paper's experiments designed and evaluated?
- `q023`: 3 → >10 — What method or technical approach does the target paper propose?
- `q010`: 2 → 9 — What limitations or unresolved issues are reported in the target paper?

Every query in the JSON artifact retains all initial Top-30 candidates with pre-rerank rank/score, rerank score, post-rerank rank, provider/model, latency, fallback, failure reason, and API request count.

## Decision

Keep `RERANK_ENABLED=false`; the evaluated Jina v3 configuration did not satisfy every production acceptance condition.

The decision compares Jina against no-rerank on paper-scoped Hit@1/MRR, NDCG@10, Recall@10, total P95, failures, fallbacks, and the number of improved queries. A disabled recommendation means `RERANK_ENABLED=false` remains the correct production setting; it is not an engineering failure.
