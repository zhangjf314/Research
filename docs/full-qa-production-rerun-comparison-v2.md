# Full QA Production Rerun Comparison v2

## Scope

- Trigger: q019 deterministic remediation and q019 single live QA passed
- Dataset: `gold-dev-v1`
- Items attempted: `50`
- LLM provider/model: `siliconflow` / `Qwen/Qwen3-8B`
- Reranker: `disabled`
- Deep Research executed: `false`
- Previous backup: `artifacts/full-qa-rerun-backups/pre-q019-validated-rerun-20260719-164745`
- Comparison JSON: `data/evaluation/full-qa-production-rerun-comparison-v2.json`

## Gate result

- Current status: `COMPLETED_WITH_FAILURES`
- Production Full QA gate: `COMPLETED_WITH_FAILURES`
- Completed/failed: `49 / 1`
- Previous failed: `q019`
- Current failed: `q024`

q019 now completes in the full 50-item rerun. The single remaining failure is q024, which failed strict citation page validation after one provider request.

## Metric comparison

| Metric | Previous | Current | Delta |
|---|---:|---:|---:|
| required_claim_coverage | 0.432624 | 0.425532 | -0.007092 |
| citation_precision | 0.101917 | 0.117908 | +0.015991 |
| citation_recall | 0.126241 | 0.153901 | +0.02766 |
| gold_block_retrieved_rate | 0.234043 | 0.276596 | +0.042553 |
| answerable_accuracy | 1.0 | 1.0 | 0 |
| refusal_accuracy | 1.0 | 1.0 | 0 |
| total_tokens | 344727 | 356378 | +11651 |
| api_requests | 50 | 50 | 0 |

Latency changed from mean/p50/p95 `25502.632 / 23382.382 / 47913.255 ms` to `23373.859 / 22314.21 / 46452.421 ms`.

## q019 status after remediation

- Status: `COMPLETED`
- Gold block present: `true`
- Citation precision / recall: `0.5 / 0.5`
- API requests: `1`
- Retry reasons: `[]`

The q019 deterministic fix is therefore validated in both the single live QA call and this full 50-item rerun.

## q024 current failure

- Status: `FAILED`
- Stage: `CLAIM_CITATION_VALIDATE`
- Error code: `CLAIM_QA_CITATION_VALIDATION_ERROR`
- Retry reasons: `citation_validation:page`
- API request count: `1`
- Wall ms: `20814.644`

## Interpretation

The q019 deterministic fix improved the input path and q019 no longer blocks Full QA. Overall citation precision, citation recall, and gold-block retrieved rate improved relative to the prior rerun. However, the Full QA gate remains failed because q024 now has a strict citation page validation failure.

Production Deep Research remains blocked until a future authorized remediation/rerun clears the Full QA gate.

