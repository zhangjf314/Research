# Evaluation Report v1

> Provisional RC evidence only: 0/50 items have been approved by a human reviewer.

- Run time: `2026-07-13T09:42:55.833422+00:00`
- Dataset: `gold-set-v1-pending-review` (50 items, 48 answerable)
- Seed: `42`
- Embedding: `HashEmbeddingProvider(dimensions=384)`
- Reranker: `LexicalReranker`
- Answer provider: `extractive-top-context baseline`

| Variant | Hit@1 | Hit@5 | Recall@5 | Recall@10 | MRR | NDCG@10 | Faithfulness | Citation Coverage | Citation Correctness | Unsupported Claim Rate | No-answer Accuracy | Mean ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_chunk_dense | 0.292 | 0.458 | 0.458 | 0.604 | 0.368 | 0.423 | 0.963 | 0.059 | 0.067 | 0.037 | 0.960 | 1.762 | 2.360 |
| structural_chunk_dense | 0.771 | 0.875 | 0.854 | 0.854 | 0.805 | 0.814 | 0.996 | 0.785 | 0.108 | 0.004 | 0.960 | 1.923 | 2.215 |
| structural_chunk_sparse | 0.708 | 0.812 | 0.812 | 0.812 | 0.757 | 0.771 | 0.981 | 0.629 | 0.229 | 0.019 | 0.960 | 4.856 | 5.293 |
| structural_chunk_hybrid | 0.771 | 0.875 | 0.854 | 0.896 | 0.802 | 0.822 | 0.993 | 0.782 | 0.171 | 0.007 | 0.960 | 6.828 | 7.445 |
| structural_chunk_hybrid_rerank | 0.542 | 0.729 | 0.729 | 0.792 | 0.623 | 0.663 | 0.950 | 0.121 | 0.200 | 0.050 | 0.960 | 9.832 | 10.579 |
| hybrid_rerank_neighbor_context | 0.542 | 0.729 | 0.729 | 0.792 | 0.623 | 0.663 | 0.987 | 0.076 | 0.200 | 0.013 | 0.960 | 9.865 | 10.622 |

## Interpretation boundary

These values are computed by the script from pending silver-derived annotations. They are reproducible engineering baselines, not formal human-gold quality claims. Latency covers in-process retrieval/answer assembly and excludes model/network latency because the configured providers are deterministic local baselines.
