# Retrieval Context Optimization v1

> Reranker disabled; embedding, LLM, prompt, corpus, queries, filters, chunks, and Gold are frozen. Oracle Gold is never injected into Production contexts.

## Current implementation audit

Stage 11C used equal-weight RRF, retrieval K=20, context K=10, chunk-ID deduplication, a configured 12,000-token budget plus a binding 12,000-character rank-prefix cap (about 3,000 estimated tokens), and no structural expansion or diversity caps. Its trace did not explain expansion sources because no expansion policy existed. Gold-page hits without exact blocks therefore could not be repaired structurally.

The optimized strategy records original rank/score, expansion reason/source, deduplication, exclusions, final rank, token truncation, and per-page/per-section concentration. `max_blocks_per_*` caps structural context items because the indexed retrieval unit is a structural multi-block chunk.

## Pure retrieval/context results

| Stage | Experiment | Exact | Gold page | Any evidence | Duplication | Mean tokens | Truncation |
|---|---|---:|---:|---:|---:|---:|---:|
| top_k | retrieval20_context5 | 0.417 | 0.646 | 0.646 | 0.000 | 1781.3 | 0.000 |
| top_k | retrieval20_context8 | 0.479 | 0.708 | 0.708 | 0.000 | 2776.0 | 0.458 |
| top_k | retrieval20_context10 | 0.500 | 0.750 | 0.750 | 0.000 | 2951.0 | 0.917 |
| top_k | retrieval30_context8 | 0.479 | 0.708 | 0.708 | 0.000 | 2776.0 | 0.458 |
| top_k | retrieval30_context10 | 0.500 | 0.750 | 0.750 | 0.000 | 2951.0 | 0.917 |
| expansion | no_expansion | 0.500 | 0.750 | 0.750 | 0.000 | 2951.0 | 0.917 |
| expansion | neighbor_window_1 | 0.438 | 0.583 | 0.583 | 0.000 | 3003.4 | 1.000 |
| expansion | same_page_expansion | 0.438 | 0.562 | 0.562 | 0.000 | 3003.3 | 1.000 |
| expansion | neighbor_window_1_plus_page_cap | 0.521 | 0.688 | 0.688 | 0.000 | 3003.4 | 1.000 |
| diversity | no_cap | 0.500 | 0.750 | 0.750 | 0.000 | 2951.0 | 0.917 |
| diversity | max_2_blocks_per_page | 0.500 | 0.750 | 0.750 | 0.000 | 2918.4 | 0.833 |
| diversity | max_3_blocks_per_section | 0.479 | 0.750 | 0.750 | 0.000 | 2652.7 | 0.604 |
| diversity | page_and_section_cap | 0.479 | 0.750 | 0.750 | 0.000 | 2605.9 | 0.521 |
| hybrid_weight | dense_0.7_lexical_0.3 | 0.521 | 0.750 | 0.750 | 0.000 | 2928.1 | 0.833 |
| hybrid_weight | dense_0.5_lexical_0.5 | 0.500 | 0.750 | 0.750 | 0.000 | 2918.4 | 0.833 |
| hybrid_weight | dense_0.3_lexical_0.7 | 0.479 | 0.708 | 0.708 | 0.000 | 2932.6 | 0.854 |

## Selected strategies

- top_k: `retrieval20_context10`
- expansion: `no_expansion`
- diversity: `max_2_blocks_per_page`
- hybrid_weight: `dense_0.7_lexical_0.3`

Final candidate: `dense_0.7_lexical_0.3`.

## Real QA comparison

- baseline: answerable=0.875, claim=0.388889, exact=0.103009, page=0.265212, recall=0.096875, unsupported_rate=0.811111, P95=42170.931 ms, tokens=328706.
- final_candidate: answerable=0.934783, claim=0.362319, exact=0.110145, page=0.29058, recall=0.05942, unsupported_rate=0.877301, P95=44511.761 ms, tokens=310262.

## Acceptance decision

- Completed QA: 46/48; blocking failures: q033, q044.
- Actual API requests across the initial run and two resume attempts: 68.
- Conservative answerable accuracy with failed rows counted incorrect: 0.896.
- Exact precision changes on completed common queries: {'improved': 3, 'degraded': 3, 'unchanged': 40, 'mean_delta': 0.008868}.
- Page precision changes on completed common queries: {'improved': 7, 'degraded': 6, 'unchanged': 33, 'mean_delta': 0.020048}.
- Required-claim changes on completed common queries: {'improved': 8, 'degraded': 11, 'unchanged': 27, 'mean_delta': -0.021739}.
- Human audit pending: 30/30.
- Human audit strata: 10 semantic non-Gold, 10 same-Gold-page non-exact, and 10 unsupported/weak; all labels remain pending/null.
- Baseline latency mean/P95: 18701.054/42170.931 ms; candidate completed-row mean/P95: 21965.163/44511.761 ms.
- Baseline input/output/total tokens: 307953/20753/328706; candidate completed-row tokens: 290877/19385/310262.

| Gate | Passed |
|---|---:|
| exact_gold_block_availability_above_43_8 | true |
| gold_page_availability_at_least_70_8 | true |
| qa_exact_precision_meaningfully_above_10_6 | false |
| unsupported_rate_below_81_1 | false |
| conservative_answerable_accuracy_at_least_87_5 | true |
| p95_latency_within_25_percent | true |
| zero_final_qa_failures | false |
| improvement_not_driven_by_few_queries | false |
| human_audit_complete_without_severe_semantic_distortion | false |

Stage 11C.6 passed: **false**. Stage 11D smoke allowed: **false**.

## Decision status

Human citation audit remains pending. Automated semantic support is not treated as human evidence. Stage 11D smoke is not authorized solely by these automated diagnostics.
