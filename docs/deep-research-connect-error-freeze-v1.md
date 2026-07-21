# Deep Research ConnectError Freeze v1

Historical run `live-q003-cbc99df5b041` is preserved unchanged.

- Question: `q003`
- Commit at freeze: `391187f4bee93ff8b90b72d35d1f463d3efb54bd`
- Provider/model: `deepseek` / `deepseek-v4-flash`
- Base URL host: `api.deepseek.com`
- Status: `provider_failed`
- Failure node: `synthesize`
- Nodes completed: `plan -> retrieve -> assess_evidence`
- Retrieval calls: `1`
- Request attempts: `1`
- Provider completed requests: `0`
- Settled tokens: `0`
- Active reserved tokens in historical run: `8081`
- Cost: `0`
- Citation validation: `not_run`

The original run only persisted a top-level `ConnectError`; it did not persist the lower-level cause chain or a raw provider response. No raw provider response is fabricated because the request did not complete.

Stage 13.38 reproduction showed that the same host `.venv` Deep Research provider POST can fail in the normal Codex execution context with `DEEP_RESEARCH_SOCKET_PERMISSION_ERROR` / `WinError 10013`. Running the same short provider path with elevated execution obtained a provider response, but the first debug script version failed during local post-processing because it called a non-existent `_estimated_cost` helper. That script bug has been fixed.
