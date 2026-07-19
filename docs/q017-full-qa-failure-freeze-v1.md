# q017 Full QA Failure Freeze v1

Frozen at: 2026-07-18T15:15:00+00:00

## Outcome

q017 remains a real Production Full QA failure. The Full QA batch must stay
`COMPLETED_WITH_FAILURES`; Production Deep Research should not be started from
this result.

## Terminal failure

- Question: `q017`
- Runtime provider/model: `siliconflow` / `Qwen/Qwen3-8B`
- Prompt version: `qa-production-v1`
- Prompt SHA-256: `ea315b0812d345faa19156abf6af075571919e55ef98791c65f3b4ad6b3ae96f`
- Reranker: disabled
- Template fallback: false
- HTTP status: `503`
- Error code: `CLAIM_QA_JSON_PARSE_ERROR`
- Stage: `LLM_JSON_PARSE`
- Request ID: `d7be58fd-7b6e-4f55-aa03-1e4c74f57225`
- API request count from error body: `1`
- Retry reasons: `["malformed_json"]`
- Rate-limit events: `0`

## Input/context audit

The gold evidence is valid and exists in the container parsed artifacts:

- Paper: `2001.08361`
- Gold block: `b000033`
- Gold chunk: `cabe1d24-ba08-42c9-8550-c746d06052df`
- Section: `Abstract`
- Page: `1`

For the production query `What are the target paper's main contributions?` with
the paper filter applied, the gold chunk ranked:

- Dense rank at recall 100: `43`
- Sparse rank at recall 100: `29`
- Fusion rank at recall 100: `40`

The final QA context contained 3 chunks and did not contain `b000033`.

This means q017 has an independent retrieval/context mismatch: the answerable
gold-dev question expected the abstract contribution block, but the production
QA route supplied later Introduction/method chunks instead.

## Root-cause assessment

API wrapper anomaly: unlikely.

The API route correctly caught `LLMProviderError` and returned a structured
HTTP 503 containing the provider error code, stage, request count, retry
reasons, rate-limit count, and request ID. There is no evidence that FastAPI
wrapping or the runner converted a successful provider response into a failure.

Model malformed JSON: confirmed at the provider classification level.

The provider sent one real request and failed in `LLM_JSON_PARSE` with
`malformed_json`. However, the current failure path appends diagnostic attempts
only after `normalize_structured_qa_content(content)` succeeds. Therefore the
raw malformed model content was not persisted and cannot be inspected from this
run.

Prompt/schema contribution: likely.

`qa-production-v1` uses natural-language schema instructions and
`response_format={"type":"json_object"}`. It does not use a strict JSON-schema
transport, tool call schema, or a retry/repair pass in this Production Full QA
configuration. With retries disabled, one malformed generation is terminal.

One-off vs systematic: indeterminate with current artifacts.

This specific q017 request was run once, and the raw malformed payload is not
available. It may be a one-completion model failure. But the project already
has historical Qwen malformed-json failures in Dev v3.x runs, so it should not
be dismissed as harmless randomness.

## Freeze decision

q017 should be frozen as:

```text
terminal_failure = malformed_json
primary_failure_stage = LLM_JSON_PARSE
api_wrapper_failure = false
raw_payload_available = false
retrieval_context_gold_present = false
production_full_qa_gate = COMPLETED_WITH_FAILURES
```

## Recommended next actions

1. Do not mark Production Full QA as passed.
2. Do not start Production Deep Research from this batch.
3. Before any future live retry, persist sanitized raw model output and response
   audit even on JSON parse failures.
4. Add a no-LLM q017 retrieval/context regression so the abstract contribution
   block enters context, or document why paper-contribution questions need an
   abstract/contribution routing rule.
5. If the user later authorizes a retry, run q017 once with raw-response
   persistence and no automatic JSON repair.

