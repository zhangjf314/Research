# Retrieval Ablation v2 — Scope-aware Protocol

- Corpus: 34 included documents (33 research papers + 1 text-native release fixture)
- Excluded but retained: 2 OCR fixtures
- Protocol: 50 approved source records; query revisions are independently tracked
- Pending query revision reviews: 0
- Scope distribution: global 0, paper 46, multi_paper 2, unanswerable 2
- Reranker: disabled
- LLM calls: none
- Hash and Jina evaluation collections share the same chunk signature

## Global retrieval

No source question is a genuine paper-discovery task. Global metrics are intentionally `N/A` rather than manufacturing title-bearing queries from known answers.

## Paper-scoped block retrieval

| Variant | Block Hit@1 | Block Hit@5 | Recall@5 | Recall@10 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| hash_structural_dense | 0.000 | 0.065 | 0.033 | 0.076 | 0.030 | 0.024 |
| hash_structural_hybrid | 0.043 | 0.130 | 0.063 | 0.214 | 0.096 | 0.084 |
| jina_structural_dense | 0.087 | 0.239 | 0.138 | 0.227 | 0.148 | 0.103 |
| jina_structural_hybrid | 0.152 | 0.348 | 0.221 | 0.278 | 0.231 | 0.144 |

## Multi-paper retrieval

| Variant | Paper coverage@5 | All papers@5 | Evidence recall@5 | Paper coverage@10 | All papers@10 | Evidence recall@10 | Evidence MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| hash_structural_dense | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.100 | 0.071 |
| hash_structural_hybrid | 1.000 | 1.000 | 0.300 | 1.000 | 1.000 | 0.300 | 0.167 |
| jina_structural_dense | 1.000 | 1.000 | 0.100 | 1.000 | 1.000 | 0.700 | 0.222 |
| jina_structural_hybrid | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 0.600 | 0.667 |

## Unanswerable retrieval behavior

| Variant | Queries | Non-empty rate | Mean top score | Max top score |
|---|---:|---:|---:|---:|
| hash_structural_dense | 2 | 1.000 | 0.400 | 0.424 |
| hash_structural_hybrid | 2 | 1.000 | 0.032 | 0.032 |
| jina_structural_dense | 2 | 1.000 | 0.600 | 0.604 |
| jina_structural_hybrid | 2 | 1.000 | 0.032 | 0.033 |

These scores are descriptive only. Retrieval returning a passage is not evidence that an answer exists; refusal requires a later calibrated QA protocol.

## Latency

| Variant | Mean ms | P50 ms | P95 ms | Failures |
|---|---:|---:|---:|---:|
| hash_structural_dense | 12.462 | 6.861 | 50.185 | 0 |
| hash_structural_hybrid | 9.911 | 7.112 | 46.206 | 0 |
| jina_structural_dense | 532.320 | 482.388 | 763.114 | 0 |
| jina_structural_hybrid | 523.764 | 510.863 | 705.999 | 0 |

## Category breakdown

| Variant | Category | Answerable | Block Hit@5 | Block Recall@10 | MRR |
|---|---|---:|---:|---:|---:|
| hash_structural_dense | algorithm_steps | 5 | 0.200 | 0.167 | 0.070 |
| hash_structural_dense | experiment_results | 4 | 0.250 | 0.042 | 0.125 |
| hash_structural_dense | experiment_setup | 5 | 0.000 | 0.000 | 0.000 |
| hash_structural_dense | limitations | 7 | 0.000 | 0.214 | 0.042 |
| hash_structural_dense | method | 5 | 0.200 | 0.200 | 0.050 |
| hash_structural_dense | multi_paper_comparison | 2 | 0.000 | 0.100 | 0.071 |
| hash_structural_dense | paper_contributions | 10 | 0.000 | 0.000 | 0.000 |
| hash_structural_dense | research_background | 10 | 0.000 | 0.000 | 0.000 |
| hash_structural_dense | unanswerable | 0 | N/A | N/A | N/A |
| hash_structural_hybrid | algorithm_steps | 5 | 0.200 | 0.350 | 0.258 |
| hash_structural_hybrid | experiment_results | 4 | 0.000 | 0.042 | 0.031 |
| hash_structural_hybrid | experiment_setup | 5 | 0.200 | 0.400 | 0.083 |
| hash_structural_hybrid | limitations | 7 | 0.143 | 0.512 | 0.241 |
| hash_structural_hybrid | method | 5 | 0.000 | 0.050 | 0.025 |
| hash_structural_hybrid | multi_paper_comparison | 2 | 0.500 | 0.300 | 0.167 |
| hash_structural_hybrid | paper_contributions | 10 | 0.100 | 0.025 | 0.025 |
| hash_structural_hybrid | research_background | 10 | 0.200 | 0.183 | 0.054 |
| hash_structural_hybrid | unanswerable | 0 | N/A | N/A | N/A |
| jina_structural_dense | algorithm_steps | 5 | 0.200 | 0.300 | 0.229 |
| jina_structural_dense | experiment_results | 4 | 0.250 | 0.125 | 0.250 |
| jina_structural_dense | experiment_setup | 5 | 0.600 | 0.640 | 0.225 |
| jina_structural_dense | limitations | 7 | 0.571 | 0.429 | 0.393 |
| jina_structural_dense | method | 5 | 0.200 | 0.100 | 0.067 |
| jina_structural_dense | multi_paper_comparison | 2 | 0.500 | 0.700 | 0.222 |
| jina_structural_dense | paper_contributions | 10 | 0.100 | 0.075 | 0.033 |
| jina_structural_dense | research_background | 10 | 0.000 | 0.100 | 0.013 |
| jina_structural_dense | unanswerable | 0 | N/A | N/A | N/A |
| jina_structural_hybrid | algorithm_steps | 5 | 0.800 | 0.550 | 0.373 |
| jina_structural_hybrid | experiment_results | 4 | 0.000 | 0.042 | 0.028 |
| jina_structural_hybrid | experiment_setup | 5 | 0.600 | 0.507 | 0.358 |
| jina_structural_hybrid | limitations | 7 | 0.571 | 0.524 | 0.538 |
| jina_structural_hybrid | method | 5 | 0.200 | 0.180 | 0.120 |
| jina_structural_hybrid | multi_paper_comparison | 2 | 1.000 | 0.600 | 0.667 |
| jina_structural_hybrid | paper_contributions | 10 | 0.100 | 0.075 | 0.100 |
| jina_structural_hybrid | research_background | 10 | 0.300 | 0.200 | 0.150 |
| jina_structural_hybrid | unanswerable | 0 | N/A | N/A | N/A |

## Difficulty breakdown

| Variant | Difficulty | Answerable | Block Hit@5 | Block Recall@10 | MRR |
|---|---|---:|---:|---:|---:|
| hash_structural_dense | easy | 18 | 0.056 | 0.065 | 0.040 |
| hash_structural_dense | hard | 24 | 0.083 | 0.064 | 0.027 |
| hash_structural_dense | medium | 6 | 0.000 | 0.167 | 0.028 |
| hash_structural_hybrid | easy | 18 | 0.167 | 0.139 | 0.109 |
| hash_structural_hybrid | hard | 24 | 0.125 | 0.247 | 0.102 |
| hash_structural_hybrid | medium | 6 | 0.167 | 0.333 | 0.061 |
| jina_structural_dense | easy | 18 | 0.222 | 0.122 | 0.114 |
| jina_structural_dense | hard | 24 | 0.333 | 0.360 | 0.211 |
| jina_structural_dense | medium | 6 | 0.000 | 0.167 | 0.021 |
| jina_structural_hybrid | easy | 18 | 0.222 | 0.122 | 0.129 |
| jina_structural_hybrid | hard | 24 | 0.500 | 0.421 | 0.350 |
| jina_structural_hybrid | medium | 6 | 0.333 | 0.278 | 0.208 |

## Hash/Jina query changes

### Improvements

- `q007`: Hash rank >10, Jina rank 1 — What are the target paper's main contributions?
- `q049`: Hash rank >10, Jina rank 1 — Compare the main contributions of the two target papers.
- `q025`: Hash rank 10, Jina rank 1 — What limitations or unresolved issues are reported in the target paper?
- `q044`: Hash rank >10, Jina rank 3 — How are the target paper's experiments designed and evaluated?
- `q045`: Hash rank 9, Jina rank 1 — What limitations or unresolved issues are reported in the target paper?

### Regressions

- `q027`: Hash rank 4, Jina rank >10 — What are the target paper's main contributions?
- `q041`: Hash rank 5, Jina rank >10 — What research problem does the target paper address?
- `q035`: Hash rank 6, Jina rank >10 — What limitations or unresolved issues are reported in the target paper?
- `q040`: Hash rank 6, Jina rank 8 — What limitations or unresolved issues are reported in the target paper?
- `q009`: Hash rank 8, Jina rank 9 — How are the target paper's experiments designed and evaluated?

## v1 versus v2 protocol

v1 ran all answerable items as unrestricted paper discovery and scored paper IDs. v2 treats 46 known-paper questions as filtered within-paper block retrieval and two comparisons as filtered multi-paper evidence retrieval. Therefore the numeric MRR values below describe different tasks and must not be interpreted as a direct improvement percentage.

| Run | Hash Hybrid MRR | Jina Hybrid MRR | Meaning |
|---|---:|---:|---|
| v1 | 0.158 | 0.082 | Unrestricted paper-ID retrieval |
| v2 | 0.096 | 0.231 | Paper-filtered block retrieval |

## Decision

The measured paper-scoped acceptance condition supports Jina as the retrieval embedding candidate.
All query revisions are human-approved. Stage 11B may begin with Reranker still disabled by default; no Reranker result is inferred from this run.
