# Grounding Alignment Audit v1

This is an offline diagnostic audit of the existing canary outputs. It does not call an LLM, does not change Gold, and does not overwrite historical metrics.

## Frozen baselines

### QWEN_CANARY_BASELINE

- attempted/completed: `15` / `15`
- required_claim_coverage: `0.238095`
- legacy citation_precision: `0.244048`
- legacy citation_recall: `0.119048`
- core_unsupported_claim_count: `30`
- composite_claim_rate: `0.170732`
- atomic_claim_rate: `0.829268`

Claim coverage failure classification:

- `CLAIM_EXPRESSED_WITH_DIFFERENT_WORDING`: `8`
- `CLAIM_MATCHER_FALSE_NEGATIVE`: `2`
- `CLAIM_TOO_BROAD`: `1`
- `CLAIM_TRULY_OMITTED`: `18`

Citation exact-match failure classification:

- `CITATION_SUPPORTS_CLAIM_BUT_NOT_GOLD_BLOCK`: `33`
- `CITATION_SUPPORTS_CLAIM_ON_EQUIVALENT_PAGE`: `16`

Unsupported-claim diagnostic classification:

- `SUPPORTED_BY_EQUIVALENT_PAGE`: `8`
- `SUPPORTED_BY_NON_GOLD_BLOCK`: `22`

### DEEPSEEK_CANARY_BASELINE

- attempted/completed: `15` / `15`
- required_claim_coverage: `0.357143`
- legacy citation_precision: `0.202381`
- legacy citation_recall: `0.139286`
- core_unsupported_claim_count: `30`
- composite_claim_rate: `0.3`
- atomic_claim_rate: `0.7`

Claim coverage failure classification:

- `CLAIM_EXPRESSED_WITH_DIFFERENT_WORDING`: `7`
- `CLAIM_MATCHER_FALSE_NEGATIVE`: `4`
- `CLAIM_TRULY_OMITTED`: `16`

Citation exact-match failure classification:

- `CITATION_SUPPORTS_CLAIM_BUT_NOT_GOLD_BLOCK`: `24`
- `CITATION_SUPPORTS_CLAIM_ON_EQUIVALENT_PAGE`: `14`

Unsupported-claim diagnostic classification:

- `SUPPORTED_BY_EQUIVALENT_PAGE`: `10`
- `SUPPORTED_BY_NON_GOLD_BLOCK`: `20`

## Auditor conclusion

- The evaluator is deterministic and internally consistent, but the names `citation_precision` and `citation_recall` are broader than their actual exact-Gold-block semantics.
- The same `core_unsupported_claim_count=30` across Qwen and DeepSeek is not explained by invalid citations or JSON/schema instability; both runs had stable structured outputs.
- Offline context coverage is high, so the dominant failure mode is downstream claim/evidence selection and exact-Gold matching, not basic context absence.
- This audit is not a blind holdout and must not be used as a strong generalization claim.
