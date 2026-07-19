# Stage 13.32 Audit

## Scope

Stage 13.32 addressed q017 Claim QA observability, retrieval/context selection,
and one controlled q017 live retry. It did not run Full QA, Deep Research, or
any second q017 retry.

## Original q017 failure

- sample_id: `q017`
- original request_id: `d7be58fd-7b6e-4f55-aa03-1e4c74f57225`
- original failure stage: `LLM_JSON_PARSE`
- original raw payload available: `false`
- original context contained gold: `false`
- original Full QA status: `COMPLETED_WITH_FAILURES`

The original freeze files were not overwritten:

- `data/evaluation/q017-full-qa-failure-freeze-v1.json`
- `docs/q017-full-qa-failure-freeze-v1.md`

## Observability repair

Response audit persistence was added at provider level.

Storage:

- Private audit directory: `artifacts/private/qa-response-audits/`
- Public q017 sanitized summary: `artifacts/q017-live-retry-response-audit-sanitized-v1.json`

Saved fields include provider/model, prompt version, response format, HTTP
status, finish reason, content length, content SHA-256, sanitized prefix/suffix,
sanitized parse-error window, reasoning/tool-call presence, usage fields, parse
error type, line, column, offset, and normalization events.

Sanitization:

- API keys, Authorization headers, Cookies, database URLs, and local absolute
  paths are redacted.
- Full payload storage defaults to `false`.
- Parse replay from a truncated sanitized audit returns
  `PARSE_REPLAY_BLOCKED_BY_PARTIAL_PAYLOAD`.

## JSON result for q017 retry

- Content empty: `false`
- Markdown fence detected: `false`
- Think tag detected: `false`
- Truncation detected: `false`
- Parse error position: `null`
- JSON status: `passed`
- Schema status: `passed`
- Citation status: `passed`

The q017 retry did not reproduce the malformed JSON failure. The original
malformed payload remains unavailable and must not be reconstructed.

## Retrieval/context root cause

q017 gold evidence:

- paper: `2001.08361`
- block: `b000033`
- chunk: `cabe1d24-ba08-42c9-8550-c746d06052df`
- section: `Abstract`
- page: `1`

Original frozen ranks:

- dense rank at 100: `43`
- sparse rank at 100: `29`
- fusion rank at 100: `40`
- final context count: `3`
- gold in context: `false`

Root cause:

- Contribution intent was handled with generic similarity ranking.
- Production recall/context candidate depth was too shallow.
- Existing section prior could not see rank-40 fused evidence.
- Top-3 did not contain complete equivalent evidence for the three required claims.

Fix:

- Generic paper-scoped contribution queries now use effective recall `max(recall_k, 60)`.
- Abstract/Contributions are prioritized during context candidate selection.
- No sample_id, question_id, or block_id hardcoding was used.

Current q017:

- pre-rerank candidate count: `60`
- final context count: `4`
- gold block in context: `true`

## Controlled q017 live retry

- Command: `scripts/run_production_qa_smoke_v1.py --sample-id q017 --single-attempt --no-json-repair --no-qa-retry`
- Result: `PASSED`
- Real model called: `true`
- Provider/model: `siliconflow` / `Qwen/Qwen3-8B`
- Template fallback: `false`
- Reranker: `disabled`
- API/model call count: `1`
- JSON repair count: `0`
- QA retry count: `0`
- Tokens input/output/total: `7131` / `999` / `8130`
- Cost: `unknown`, estimated USD `null`
- Latency: QA endpoint `41877.482` ms, wall `43577.431` ms
- Citation context validity: `1.0`

## Full QA status

```text
Original Full QA status=COMPLETED_WITH_FAILURES
Full QA run in this stage=false
```

The q017 retry passing does not make Full QA pass. Because retrieval/context
configuration changed, aggregate metrics should be recalculated only through a
future explicitly authorized rerun.

## Conclusion

```text
A. q017 单次复验通过；下一步应根据配置变化决定完整重跑 Full QA。
```

