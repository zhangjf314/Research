# Evidence Retrieval v1

> Pure offline evaluation. No LLM, Reranker, Deep Research, new Embedding request, Oracle evidence, or Gold evidence was used for selection.

| Variant | Exact block | Gold page | Block recall | Multi-paper | Non-evidence | Metadata | Duplicate | Mean tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_retrieval | 0.438 | 0.708 | 0.270 | 1 | 0.734 | 0.025 | 0.104 | 3160.1 |
| evidence_unit_retrieval | 0.542 | 0.771 | 0.235 | 1 | 0.000 | 0.000 | 0.000 | 1711.4 |
| routed_evidence_retrieval | 0.646 | 0.875 | 0.319 | 1 | 0.000 | 0.000 | 0.000 | 1697.5 |
| claim_first_evidence_retrieval | 0.125 | 0.250 | 0.042 | 1 | 0.000 | 0.000 | 0.000 | 194.5 |

Best offline candidate: `routed_evidence_retrieval`.

## Dev QA gate

- exact_block_at_least_0_65: False
- gold_page_at_least_0_80: True
- metadata_below_baseline: True
- context_tokens_within_125_percent: True
- offline_p95_below_500_ms: True
- improvement_across_categories_and_difficulties: True
- citation_triple_trace_complete: True
- no_oracle_or_gold_injection: True
- reranker_disabled: True
- llm_not_called: True
- no_new_embedding_requests: True

Dev QA authorized by offline gate: **False**.

Claim evidence set recall and all-required-claims evidence availability are intentionally null until the 146 claim-level mappings are reviewed. Question-level Gold is used only after selection to compute exact block/page metrics.

Detailed paper/multi-paper/category/difficulty slices and per-query traces are stored in the JSON artifact.
