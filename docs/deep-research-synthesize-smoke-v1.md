# Deep Research Synthesize Smoke v1

`inspect` mode passed and did not call the LLM. It confirmed:

- Run: `live-q003-cbc99df5b041`
- Provider/model from shared factory: `deepseek` / `deepseek-v4-flash`
- Client: sync `httpx.Client`
- Prompt version: `qa-production-v1`
- Context count: available from persisted q003 retrieval context
- API key: present; only length and SHA-256 prefix persisted

The first normal host provider-smoke failed before completion with:

- Classification: `DEEP_RESEARCH_SOCKET_PERMISSION_ERROR`
- WinError: `10013`
- Host: `api.deepseek.com`
- Port: `443`

An elevated host provider-smoke obtained a provider response but failed during local audit post-processing because the debug script tried to call a non-existent `_estimated_cost` helper. The script has been corrected to reuse provider `_usage()`.

After explicit authorization, the corrected provider-smoke was run once and passed:

- Status: `PASSED`
- HTTP status: `200`
- Request count: `1`
- JSON valid: `true`
- Input/output/total tokens: `53` / `20` / `73`
- Estimated cost: `$0.00001302`
- Raw payload persisted: `false`
- API key persisted: `false`
- Authorization header persisted: `false`
