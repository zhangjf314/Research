# Portfolio Evidence Manifest v1

- Generated at: `2026-08-09`
- Branch before final merge: `eval/research-agent-benchmark-run-v1`
- Stage 4C.1 audit commit: `dd7401b530eee98658c93033cd0f57d58a344587`
- Version: `1.1.0+portfolio` / display `1.1.0-portfolio`
- Target: `v1.1.0-portfolio`
- Release decision from this manifest: `PORTFOLIO_RELEASE_FINALIZATION_AUTHORIZED`

## Frozen evaluation evidence

Full QA evidence was not rerun in Stage 13.39.

- File: `data/evaluation/deepseek-full-qa-final-v1.json`
- SHA-256: `d3b3687a8ef88c56b16e568a1bcc73198935ddff80b18b953dac584529c89a36`
- Provider/model: `deepseek` / `deepseek-v4-flash`
- Attempted/completed/failed: `50 / 50 / 0`
- Structured output success: `1.0`
- Citation ID/context/page validity: `1.0 / 1.0 / 1.0`
- Template fallback count: `0`
- Tokens: `522507 input`, `6903 output`, `529410 total`
- Estimated cost: `$0.07508382`
- P95 latency: `8216.745 ms`

Deep Research evidence was not rerun in Stage 13.39.

- Run ID: `live-q003-ed900ef2e202`
- File: `artifacts/deepseek-production-deep-research-v2/live-q003-ed900ef2e202/result.json`
- SHA-256: `e10568808dfad1924bfe34b844d637ac920a01a3288ff77787d0b963ca8de0f5`
- Status: `completed`
- Nodes: `plan -> retrieve -> assess_evidence -> synthesize -> validate_citations -> persist_trace`
- Retrieval calls: `1`
- Provider completed requests: `1`
- Tokens: `6714 input`, `82 output`, `6796 total`
- Cost: `$0.00096292`
- Citation validation: `passed`
- Reranker called: `false`

## Boundary

The QA, Deep Research, operations, and Stage 4 Workflow vs Agent engineering
evidence is sufficient for `v1.1.0-portfolio` portfolio release finalization
with explicit limitations.

`STRONG_GROUNDING_CLAIM_ALLOWED=false` and
`SEMANTIC_CLAIM_SUPPORT_AUDIT=NOT_FORMALLY_VALIDATED` remain in force.
