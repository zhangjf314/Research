# Full QA Production Rerun Comparison v1

Generated after the Stage 13.32 context-selection change and one authorized 50-question Production Full QA rerun.

## Scope

- Dataset: `gold-dev-v1`
- Items attempted: 50
- LLM provider/model: SiliconFlow `Qwen/Qwen3-8B`
- Prompt version: `qa-production-v1`
- Reranker: disabled
- Deep Research: not executed
- Previous artifacts backup: `artifacts/full-qa-rerun-backups/pre-stage13-32-rerun-20260719-001410`
- Current comparison JSON: `data/evaluation/full-qa-production-rerun-comparison-v1.json`

## Gate result

`production_full_qa_gate=COMPLETED_WITH_FAILURES`

The rerun improved the q017 failure that motivated the context-selection fix. q017 now completes and retrieves/cites the required gold block. The single current failure is q019, which failed strict citation page validation after one provider request.

This means the new context-selection configuration improved the formal metrics, but Production Full QA is still not passed.

## Metric comparison

| Metric | Previous | Current | Delta |
|---|---:|---:|---:|
| attempted | 50 | 50 | 0 |
| completed | 49 | 49 | 0 |
| failed | 1 | 1 | 0 |
| answerable_accuracy | 0.978723 | 1.0 | +0.021277 |
| refusal_accuracy | 1.0 | 1.0 | 0 |
| required_claim_coverage | 0.404255 | 0.432624 | +0.028369 |
| citation_id_validity | 1.0 | 1.0 | 0 |
| citation_precision | 0.072163 | 0.101917 | +0.029754 |
| citation_recall | 0.089007 | 0.126241 | +0.037234 |
| claim_citation_binding_rate | 1.0 | 1.0 | 0 |
| unsupported_claim_count | 169 | 166 | -3 |
| gold_block_retrieved_rate | 0.191489 | 0.234043 | +0.042554 |
| total_tokens | 356034 | 344727 | -11307 |
| api_requests | 50 | 50 | 0 |
| rate_limit_events | 0 | 0 | 0 |

Latency changed from mean/p50/p95 `26079.573 / 24665.719 / 48892.2 ms` to `25502.632 / 23382.382 / 47913.255 ms`.

## q017 status

- Previous status: failed
- Current status: completed
- Current citation recall: 1.0
- Current citation precision: 0.142857
- `gold_block_present=true`

The context-selection change therefore addressed the q017 production QA input problem without enabling reranking or changing the gold data.

## q019 current failure

- Question: `q019`
- Status: failed
- Stage: `CLAIM_CITATION_VALIDATE`
- Provider error code: `CLAIM_QA_CITATION_VALIDATION_ERROR`
- Failure reason: `citation_validation:page`
- API request count: 1
- Rate-limit events: 0
- Request ID: `03a7e352-e7e2-40e9-a04c-8931c7f2dd3f`
- Audit evidence: `artifacts/private/qa-response-audits/q019-full-qa-rerun-q019-1784391762.json`

The q019 failure is not a JSON parsing failure. The provider returned content, usage, and a completed HTTP response; strict citation validation rejected at least one cited page. This failure must remain a strict failure unless a separate root-cause pass authorizes a deterministic protocol or context fix.

## Interpretation

The rerun supports three narrow conclusions:

1. The q017 context-selection remediation worked for the original failing item.
2. Overall retrieval and citation metrics improved under the new configuration.
3. Full QA is still blocked by q019, so Production Deep Research should remain blocked.

It does not support a v1.0 or production-ready claim.

