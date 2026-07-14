# QA Context Diagnostics v1

> Status: diagnostic evidence only. Oracle modes are explicitly marked `oracle=true`, `production_metric=false`, and are excluded from Production metrics.

## Frozen protocol

- Corpus, chunks, Jina embedding, Structural Hybrid retrieval, filters, and queries are unchanged from Stage 11C.
- Reranker is disabled. The LLM remains SiliconFlow `Qwen/Qwen3-8B`, prompt `qa-production-v1`, temperature 0.
- `retrieved` reuses the frozen Stage 11C answers and makes no LLM request. The other modes are controlled Oracle diagnostics.
- Deep Research was not run. The original `qa-production-v1.*` artifacts were not overwritten.

## Metric definitions

- **Exact**: cited `(paper_id, block_id)` is in the Gold block set.
- **Page**: Exact, or the citation is on a Gold page in the same Gold paper.
- **Adjacent**: Page support, or block number distance from a Gold block is at most 2.
- **Semantic**: the preceding classes, or token-set recall from cited block to claim is at least 0.35. This is a lexical diagnostic proxy, not a human entailment judgment.
- **Recall**: Gold block identifiers cited by the answer divided by all Gold block identifiers.
- **Unsupported**: claim-level exact-Gold miss rate, retained for comparison with the strict Stage 11C criterion.

## Results

| Mode | N | Answerable | Claim coverage | Exact | Page | Adjacent | Semantic | Recall | Unsupported | Tokens | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| retrieved | 48 | 0.875 | 0.389 | 0.106 | 0.265 | 0.282 | 0.816 | 0.097 | 0.811 | 320114 | 42170.931 |
| oracle_gold_only | 48 | 0.958 | 0.542 | 0.958 | 0.958 | 0.958 | 0.958 | 0.619 | 0.000 | 65231 | 29900.371 |
| oracle_gold_plus_distractors | 48 | 1.000 | 0.569 | 0.815 | 0.848 | 0.848 | 0.981 | 0.586 | 0.220 | 246446 | 33176.493 |
| retrieved_plus_missing_gold | 48 | 0.979 | 0.486 | 0.423 | 0.589 | 0.589 | 0.934 | 0.349 | 0.561 | 365964 | 38947.521 |

## Bottleneck diagnosis

- **Retrieval evidence availability:** exact Gold blocks occur in only 43.8% of retrieved contexts; Gold pages occur in 70.8%. Gold-only Oracle raises answerable accuracy from 87.5% to 95.8% and required-claim coverage from 38.9% to 54.2%.
- **Context distraction:** adding distractors to the same Gold evidence lowers exact precision from 95.8% to 81.5% and raises strict unsupported rate from 0.0% to 22.0%.
- **Appending missing Gold is not sufficient:** it raises answerable accuracy to 97.9%, but exact precision remains 42.3% and strict unsupported rate 56.1%; the original distractors still dominate the context.
- **Gold exactness is narrow:** retrieved exact/page/adjacent/semantic precision is 10.6%/26.5%/28.2%/81.6%. Semantic support is only an automated lexical proxy and remains pending human audit.
- **LLM is not a perfect upper bound:** even Gold-only context reaches 95.8% answerable accuracy and 54.2% claim coverage, so generation/claim selection remains a secondary bottleneck after retrieval and context selection.

The primary next step is retrieval and context selection optimization, followed by a small human citation audit. These Oracle results do not justify a Production or v1.0 claim.

## Stage 11C refusal recovery

| Question | Recovered after adding missing Gold | Claim coverage | Exact precision |
|---|---:|---:|---:|
| q022 | true | 0.333 | 1.000 |
| q023 | false | 0.000 | 0.000 |
| q024 | true | 0.000 | 0.000 |
| q037 | true | 0.667 | 1.000 |
| q043 | true | 0.000 | 0.000 |
| q047 | true | 0.667 | 1.000 |

Five of the six answerable Stage 11C refusals recovered (5/6); q023 remained a refusal. Recovery alone is not correctness: q024 and q043 still have zero required-claim coverage and zero exact precision.

## Citation classifications and unsupported categories

### `retrieved`

Classifications: `{"adjacent_to_gold_block": 5, "exact_gold_block": 34, "same_gold_page": 29, "semantic_support_non_gold": 108, "unsupported": 2, "weakly_related": 8}`

Unsupported categories: `{"citation_not_supporting_claim": 10, "context_support_but_not_gold": 136, "extra_claim_not_in_required_claims": 121, "required_claim_semantic_miss": 88}`

### `oracle_gold_only`

Classifications: `{"exact_gold_block": 167}`

Unsupported categories: `{"extra_claim_not_in_required_claims": 71, "required_claim_semantic_miss": 66}`

### `oracle_gold_plus_distractors`

Classifications: `{"exact_gold_block": 141, "same_gold_page": 3, "semantic_support_non_gold": 33, "weakly_related": 4}`

Unsupported categories: `{"citation_not_supporting_claim": 3, "context_support_but_not_gold": 36, "extra_claim_not_in_required_claims": 85, "required_claim_semantic_miss": 62}`

### `retrieved_plus_missing_gold`

Classifications: `{"exact_gold_block": 89, "same_gold_page": 28, "semantic_support_non_gold": 77, "unsupported": 2, "weakly_related": 5}`

Unsupported categories: `{"citation_not_supporting_claim": 7, "context_support_but_not_gold": 104, "extra_claim_not_in_required_claims": 117, "required_claim_semantic_miss": 74}`

## Compatibility note

Stage 11C reports 147 unsupported claims. This diagnostic's claim-level exact-Gold recount is 146. The sole difference is q004: two claims cite the same exact Gold block `b000103`; Stage 11C's aggregate counts that unique supporting citation once, while this diagnostic credits both bound claims. No answer or Gold record changed.

## Known limitations

- Oracle context proves evidence availability effects, not a deployable retrieval policy.
- Token-set recall can over-credit lexically similar but non-entailing evidence; no row has been represented as human-approved.
- Exact Gold annotations may omit valid same-page, adjacent, or alternative evidence blocks.
- Latency and token totals for `retrieved` are inherited from the frozen Stage 11C run; Oracle modes are new live calls and are not directly cost-equivalent.
- The experiment covers 48 answerable questions. The two unanswerable questions are intentionally excluded from these context-recall diagnostics.
