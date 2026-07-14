# Citation Human Audit Summary v1

> AI-assisted manual citation audit, 30-sample stratified review. This is not an independent blind review and does not confirm full-dataset human citation precision.

## Validation

- Records: 30/30; approved: 30; pending: 0; unique sample IDs and claim-citation pairs: 30.
- Reviewer, reviewed_at, and review_notes are complete. Claim, citation, Gold evidence, and automated labels are unchanged.

## Human support rates

- Strict support (`fully_supported` only): 5/30 = 16.7%.
- Lenient support (`fully_supported` + `partially_supported`): 7/30 = 23.3%.
- Related but insufficient: 7/30 = 23.3%.
- Fully unsupported: 16/30 = 53.3%.
- Gold annotation too narrow: 0/30 = 0.0%.

## Automated label calibration

| Automated label | N | Strict precision | Lenient precision | Human unsupported |
|---|---:|---:|---:|---:|
| `exact_gold_block` | 0 | N/A | N/A | 0 |
| `same_gold_page` | 10 | 20.0% | 20.0% | 6 |
| `semantic_support_non_gold` | 10 | 30.0% | 40.0% | 2 |
| `weakly_related` | 8 | 0.0% | 12.5% | 6 |
| `unsupported` | 2 | 0.0% | 0.0% | 2 |

## Confusion matrices

- Strict: `{"false_negative": 0, "false_positive": 15, "true_negative": 10, "true_positive": 5}`
- Lenient: `{"false_negative": 1, "false_positive": 14, "true_negative": 9, "true_positive": 6}`

## Findings

- Token-set semantic support strict/lenient precision: 30.0%/40.0%; lenient false positives: 6.
- Same-page samples judged fully unsupported: 6.
- Weakly-related samples judged fully unsupported: 6.
- Automated unsupported negative precision: 100.0%.
- Gold annotation too narrow: 0.

The prior 81.6% semantic-support value is a token-overlap signal, not citation correctness. In this stratified sample, only 5 citations are fully supported and 7 are fully or partially supported; 23 are related-but-insufficient or unsupported. The automated semantic label materially overestimates human-confirmed support.
