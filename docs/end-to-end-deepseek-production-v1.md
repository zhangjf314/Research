# End-to-End DeepSeek Production Deep Research v1

Status: `FAILED_PROVIDER_CONNECT_ERROR`

This Stage 13.37 run was started only after the 50-item DeepSeek Direct Full QA engineering gate passed. It did not complete the Deep Research graph.

## Run

- Run ID: `live-q003-cbc99df5b041`
- Question ID: `q003`
- Provider/model: `deepseek` / `deepseek-v4-flash`
- Prompt version: `qa-production-v1`
- Billing mode: `paid`
- Cost basis: `configured_token_prices`
- Reranker called: `false`
- Template fallback: `false`
- Output root: `artifacts/deepseek-production-deep-research-v1`

## Result

- Graph status: `provider_failed`
- Nodes completed before failure: `plan -> retrieve -> assess_evidence`
- Current node at failure: `synthesize`
- Retrieval calls: `1`
- LLM request attempts: `1`
- Provider completed requests: `0`
- Usage records: `0`
- Settled tokens: `0`
- Active reserved tokens: `8081`
- Monetary cost USD: `0`
- Citation validation: `not_run`
- Failure type: `ConnectError`
- Failure message: `deepseek QA failed after 1 request(s): ConnectError`

The failure occurred after request reservation and request start, before any provider-reported usage was returned. The run evidence is preserved in the isolated run directory and must not be replaced with a passing claim.

## Gate conclusion

- `DEEPSEEK_DIRECT_FULL_QA_ENGINEERING_GATE=PASSED`
- `DEEPSEEK_PRODUCTION_DEEP_RESEARCH_GATE=FAILED`
- Stage 13.37 final conclusion: `B. 50 QA Engineering Gate passed, Deep Research not yet passed.`

## Known warnings recorded

- Qdrant HTTP connection carries an API key over an insecure connection warning.
- qdrant-client `1.18.0` reports a compatibility warning against Qdrant server `1.12.5`.
