# DeepSeek Provider Smoke Audit v1

- Status: `PASSED`
- Provider/model: `deepseek` / `deepseek-v4-flash`
- Base URL host: `api.deepseek.com`
- Thinking: `disabled`
- Response format: `json_object`
- Stream: `false`
- Template fallback: `false`
- API key persisted: `false`
- Authorization header persisted: `false`

## Codex host preflight

- DNS: `passed`
- TCP 443: `passed`
- TLS/HTTPS: `passed`
- `/models`: `passed`
- Minimal completion: `passed`
- Minimal completion model: `deepseek-v4-flash`
- Minimal completion finish reason: `stop`
- Reasoning content present: `false`
- safe_to_start_batch: `true`

## Docker API container preflight

- DNS: `passed`
- TCP 443: `passed`
- TLS/HTTPS: `passed`
- `/models`: `passed`
- Minimal completion: `passed`
- Minimal completion model: `deepseek-v4-flash`
- Minimal completion finish reason: `stop`
- Reasoning content present: `false`
- safe_to_start_batch: `true`

## q024 claim-contract smoke

- Status: `PASSED`
- Real model called: `true`
- Provider/model: `deepseek` / `deepseek-v4-flash`
- JSON parse: `passed`
- Schema: `passed`
- Citation validation: `passed`
- Page validation: `passed`
- Provider call count: `1`
- QA retry count: `0`
- JSON repair count: `0`
- Citation repair count: `0`

No full API key or Authorization header is recorded in this audit.
