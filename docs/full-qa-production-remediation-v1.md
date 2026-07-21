# Full QA Production Remediation v1

## Original Full QA state

- Original status: `COMPLETED_WITH_FAILURES`
- Original failed sample: `q017`
- Original request ID: `d7be58fd-7b6e-4f55-aa03-1e4c74f57225`
- Original failure stage: `LLM_JSON_PARSE`
- Original raw payload available: `false`
- Original context contained gold: `false`

The original Full QA report remains failed. This remediation record does not
rewrite `data/evaluation/full-qa-production-v1.json` or the original q017
failure freeze.

## Stage 13.32 changes

- Added sanitized provider response audit persistence for Claim QA responses.
- Added malformed JSON classification metadata.
- Added parse-only replay support for response-audit files.
- Added a generic paper-contribution context route:
  - paper-scoped contribution queries use effective recall `max(recall_k, 60)`;
  - Abstract/Contributions sections are prioritized before Introduction during
    final context candidate selection.
- No q017/question-id/block-id hardcoding was introduced.

## q017 single retry

- Command: `scripts/run_production_qa_smoke_v1.py --sample-id q017 --single-attempt --no-json-repair --no-qa-retry`
- Result: `PASSED`
- Real provider: `true`
- Provider/model: `siliconflow` / `Qwen/Qwen3-8B`
- Template fallback: `false`
- API request count: `1`
- JSON repair count: `0`
- QA retry count: `0`
- Input/output/total tokens: `7131` / `999` / `8130`
- Estimated cost USD: `null` (`cost_status=unknown`)
- QA endpoint latency ms: `41877.482`
- Citation context validity: `1.0`
- q017 context contained gold after fix: `true`

## Full QA handling

The single q017 retry did not convert Full QA to PASS.

An authorized 50-item Production Full QA rerun was executed after the
context-selection change.

- Rerun status: `COMPLETED_WITH_FAILURES`
- Attempted/completed/failed: `50 / 49 / 1`
- q017 current status: `COMPLETED`
- New failed sample: `q019`
- q019 failure stage: `CLAIM_CITATION_VALIDATE`
- q019 failure reason: `citation_validation:page`
- q019 API request count: `1`
- Rerun comparison: `data/evaluation/full-qa-production-rerun-comparison-v1.json`
- Rerun comparison report: `docs/full-qa-production-rerun-comparison-v1.md`

The rerun demonstrates that q017 was remediated by the generic context-selection
change, but the Production Full QA gate remains failed because q019 now fails
strict citation validation. Production Deep Research remains blocked.

Until then:

```text
PRODUCTION_FULL_QA_STATUS=COMPLETED_WITH_FAILURES
```
