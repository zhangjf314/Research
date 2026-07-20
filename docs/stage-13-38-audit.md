# Stage 13.38 Audit

## Status

- 50 DeepSeek QA Engineering Gate: `PASSED`
- Full QA rerun in this stage: `false`
- Semantic claim support audit: `NOT_FORMALLY_VALIDATED`
- Strong grounding claim allowed: `false`
- Deep Research final retry executed: `false`

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

The corrected synthesize provider-smoke has not yet been re-run after the local script fix. Therefore the single full Deep Research retry is not executed in this turn.

Final Stage 13.38 conclusion for this turn:

`C. Root cause is localized, but the corrected preflight evidence is incomplete; prohibit the final paid Deep Research retry until one additional corrected short synthesize provider-smoke is explicitly authorized.`
