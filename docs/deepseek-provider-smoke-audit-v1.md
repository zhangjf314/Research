# DeepSeek Provider Smoke Audit v1

- Status: `BLOCKED`
- Provider/model: `deepseek` / `deepseek-v4-flash`
- Base URL host: `api.deepseek.com`
- Thinking: `disabled`
- Response format: `json_object`
- Stream: `false`
- Template fallback: `false`
- API key persisted: `false`
- Authorization header persisted: `false`

## Preflight result

- DNS: `passed`
- TCP: `failed`
- TLS: `not_run`
- Models endpoint: `not_run`
- Minimal completion: `not_run`
- Error type: `PermissionError`
- safe_to_start_batch: `false`

The DeepSeek request contract was built with `model=deepseek-v4-flash`,
`response_format.type=json_object`, `thinking.type=disabled`, and `stream=false`.
No full API key or Authorization header is recorded in this audit.

## Decision

`q024` claim-contract smoke and the 15-item DeepSeek Canary were not run because
the provider preflight did not pass. This preserves the Stage 13.35 rule that
Canary cannot start before a successful low-cost DeepSeek preflight.
