# Deep Research Final Retry Config v1

The single allowed final retry is fixed as:

```powershell
.\.venv\Scripts\python.exe scripts\run_deep_research_smoke_v1.py `
  --mode live `
  --question-id q003 `
  --attempt-number 2 `
  --output-root artifacts\deepseek-production-deep-research-v2 `
  --max-cost-usd 0.2 `
  --max-total-tokens 40000 `
  --max-total-requests 4 `
  --max-total-seconds 300
```

Fixed configuration:

- Question: `q003`
- Provider/model: `deepseek` / `deepseek-v4-flash`
- Reranker: `false`
- Budget: `$0.20`, `40000` tokens, `4` requests, `300` seconds
- Provider retries: `0`
- Prompt version: `qa-production-v1`
- Output root: `artifacts/deepseek-production-deep-research-v2`

This retry must not reuse `live-q003-cbc99df5b041`.
