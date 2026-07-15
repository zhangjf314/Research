# Dev v3 Prompt Delivery Audit

- Runs/slots delivered: 10/27
- `response_format=json_object`: sent and produced valid JSON 10/10
- Complete v3 schema in system prompt: **No**
- Explicit anti-wrapper / anti-claim-map / anti-legacy rules: **No / No / No**
- Complete examples: **No**
- Historical conflict: **Yes** — the unanswerable instruction used legacy `claims=[]`.
- Exact request body/prompt hash was not persisted before Stage 13.5; reconstruction uses frozen runner code and per-run input.
- `json_schema` and tools/functions were not sent or verified.
