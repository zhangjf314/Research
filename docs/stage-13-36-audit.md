# Stage 13.36 Audit

- Generated at: `2026-07-20T11:04:38.534744+00:00`
- Git commit: `f983d3432167edf49d18cff55f6cc89a1d2230e9`
- LLM calls during offline audit: `0`
- Frozen Qwen baseline: `QWEN_CANARY_BASELINE`
- Frozen DeepSeek baseline: `DEEPSEEK_CANARY_BASELINE`
- Evaluator conclusion: `EVALUATOR_BASICALLY_VALID_BUT_EXACT_GOLD_METRIC_NAMES_ARE_TOO_BROAD`
- Recommended next step: `stop Production QA line if Evidence-first canary fails quality`
- Full QA status: `blocked`
- Deep Research status: `not run`
- Reranker status: `disabled`

This is an internal development/canary audit, not a blind holdout.

## Evidence-first Canary v1

- engineering gate: `FAILED`
- quality gate: `FAILED`
- attempted/completed/failed: `6` / `4` / `2`
- malformed/schema/invalid citation: `0` / `2` / `2`
- required_claim_coverage: `0.555556`
- citation_precision: `0.176808`
- citation_recall: `0.6`
- core_unsupported_claim_count: `89`
- budget_violations: `[]`

Conclusion: `Evidence-first engineering and quality gates failed; do not run 50-item Full QA or Deep Research.`
