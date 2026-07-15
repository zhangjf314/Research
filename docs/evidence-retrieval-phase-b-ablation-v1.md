# Evidence Retrieval Phase B Ablation v1

> Offline only. Dev QA was not run. No LLM, reranker, Deep Research, or new embedding request was made.

| Variant | Exact block | Gold page | Block recall | Claim evidence recall | Mean tokens | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| stage13_routed_baseline | 0.645833 | 0.875000 | 0.319444 | 0.384615 | 1697.50 | 23.597200 |
| phase_b_adjacent_same_page_completion | 0.729167 | 0.875000 | 0.421528 | 0.487179 | 2314.66 | 31.300000 |

- Hit gains: `['q002', 'q007', 'q013', 'q050']`
- Hit losses: `[]`
- Offline candidate gates: `{"citation_triple_trace_complete": true, "claim_evidence_set_recall_available": true, "context_tokens_within_150_percent_of_routed": true, "deep_research_not_called": true, "exact_block_at_least_0_65": true, "gold_page_at_least_0_80": true, "llm_not_called": true, "metadata_not_above_baseline": true, "no_exact_hit_regressions": true, "no_new_embedding_requests": true, "no_oracle_or_gold_injection": true, "offline_p95_below_500_ms": true, "reranker_disabled": true}`
- Allow Dev QA: **True**
- Actual Dev QA: **False**
