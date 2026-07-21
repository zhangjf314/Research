# Evidence-first Canary Audit v1

This run uses a fixed six-item subset and is not a blind holdout. It does not run Full QA or Deep Research.

## Configuration

- provider/model: `deepseek` / `deepseek-v4-flash`
- reranker: `disabled`
- concurrency: `1`
- JSON repair / QA retry / citation repair: `false` / `false` / `false`
- samples: `['q014', 'q020', 'q024', 'q001', 'q049', 'q005']`

## Direct QA baseline on same six samples

- required_claim_coverage: `0.266667`
- citation_precision: `0.266667`
- citation_recall: `0.19`
- core_unsupported_claim_count: `10`

## Evidence-first result

- engineering gate: `FAILED`
- quality gate: `FAILED`
- attempted/completed/failed: `6` / `4` / `2`
- malformed/schema/invalid citation: `0` / `2` / `2`
- required_claim_coverage: `0.555556`
- citation_precision: `0.176808`
- citation_recall: `0.6`
- core_unsupported_claim_count: `89`
- tokens: `79603` / `6281` / `85884`
- estimated_cost_usd: `0.0129031`
- budget_violations: `[]`

Evidence-first did not satisfy the quality gate; Full QA remains blocked.
