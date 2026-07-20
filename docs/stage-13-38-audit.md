# Stage 13.38 Audit

## Status

- 50 DeepSeek QA Engineering Gate: `PASSED`
- Full QA rerun in this stage: `false`
- Semantic claim support audit: `NOT_FORMALLY_VALIDATED`
- Strong grounding claim allowed: `false`
- Deep Research final retry executed: `true`

## Root cause evidence

The historical Deep Research failure `live-q003-cbc99df5b041` froze only a top-level `ConnectError`; it did not preserve a lower-level cause chain.

Stage 13.38 reproduced the host `.venv` provider POST failure in the normal Codex execution context as:

- `DEEP_RESEARCH_SOCKET_PERMISSION_ERROR`
- `WinError 10013`
- Host: `api.deepseek.com`
- Port: `443`

The same provider path under elevated execution obtained a provider response, proving that the earlier failure was not a DeepSeek model/schema failure. The elevated smoke then failed in local audit post-processing due to a missing helper method; this has been fixed.

## Fixes implemented

- Deep Research smoke accepts DeepSeek through `openai_compatible` + `LLM_PROVIDER_NAME=deepseek`.
- Deep Research uses the shared `build_llm_provider` factory.
- Provider failure audit now records sanitized exception classification and cause chain.
- Provider failure and missing usage release token reservations.
- Token ledger events now include reservation, settlement, and release events.
- Added `scripts/debug_deep_research_synthesize_v1.py`.

## Gate decision

The corrected synthesize provider-smoke was explicitly authorized and passed. The single q003 attempt 2 final retry was then executed once.

## Final Deep Research result

- Run ID: `live-q003-ed900ef2e202`
- Parent run ID: `live-q003-cbc99df5b041`
- Status: `completed`
- Nodes: `plan -> retrieve -> assess_evidence -> synthesize -> validate_citations -> persist_trace`
- Retrieval calls: `1`
- Request attempts: `1`
- Provider completed requests: `1`
- Usage records: `1`
- Tokens input/output/total: `6714` / `82` / `6796`
- Cost: `$0.00096292`
- Elapsed seconds: `1.337596`
- Claims/citations: `2` / `2`
- Citation validation: `passed`
- Active reserved tokens: `0`
- Reranker called: `false`
- Template fallback: `false`

Final Stage 13.38 conclusion for this turn:

`A. Deep Research has passed and the project can proceed to safety audit, soak, and Portfolio release closure.`
