# q024 Live QA Failure Freeze v1

Status: `PASSED`

The q024 single live QA was rerun after deterministic retrieval/context remediation and schema-output hardening. It now passes strict citation validation. The 50-item Full QA rerun was not started in this step.

## Command

```powershell
.\.venv\Scripts\python.exe scripts\run_production_qa_smoke_v1.py --sample-id q024 --single-attempt --no-json-repair --no-qa-retry
```

## Final live result

- Sample: `q024`
- Status: `PASSED`
- Real model called: `true`
- Provider/model: `siliconflow` / `Qwen/Qwen3-8B`
- Prompt version: `qa-production-v1`
- Answerable: `true`
- Claim count: `3`
- Citation count: `3`
- Citation context validity: `1.0`
- All citations in retrieved context: `true`
- Citation pages: `10`, `10`, `11`
- Token usage: input `4863`, output `324`, total `5187`
- Latency: retrieval `1680.897 ms`, QA `13354.186 ms`, wall `15291.113 ms`
- Reranker: disabled
- JSON repair: disabled
- QA retry: disabled
- Citation repair: disabled
- Deep Research: not run

## Failure history and root cause

The earlier q024 failures were not caused by retrieval, context page mapping, citation page mapping, provider timeout, or API wrapper corruption.

Observed failure sequence:

1. `CLAIM_QA_SCHEMA_VALIDATION_ERROR` at `CLAIM_SCHEMA_VALIDATE`: the provider returned a response that failed strict structured QA schema validation.
2. `CLAIM_QA_UNEXPECTED_ERROR` at `API_QA_UNEXPECTED`: API exception audit captured `AttributeError: 'str' object has no attribute 'get'` in `normalize_structured_qa_content()`, caused by a non-object entry in `claims`.
3. `CLAIM_QA_JSON_PARSE_ERROR` at `LLM_JSON_PARSE`: the provider returned HTTP 200 with `finish_reason=length`, `2048` output tokens, malformed/truncated JSON, and repeated closing `</think>` tags.

Root cause: q024 exposed production QA output-contract weakness for `Qwen/Qwen3-8B`: schema-invalid claim shapes and overly long JSON generation could fail before strict validation or be truncated. This was a model output/schema constraint issue, not a q024 retrieval or page-map issue.

## Deterministic fixes

- API/smoke observability now preserves wrapped raw error details and writes sanitized API exception audits for unexpected QA failures.
- `normalize_structured_qa_content()` now fails closed for:
  - non-list `claims`;
  - non-object claim entries;
  - non-list `citations`;
  - non-object citation entries.
- `qa-production-v1` now explicitly forbids `<think>` / `</think>` / reasoning traces and constrains output to compact JSON:
  - answer <= 80 words;
  - at most 3 claims;
  - each claim text <= 25 words;
  - at most 2 citations per claim.

No retry, JSON repair, citation repair, Gold injection, or reranker was used.

## Retrieval/context status

The retrieval/context remediation worked as intended before the QA call:

- Paper: `2005.14165`
- Paper UUID: `930bea6f-5263-4012-8451-c2d19c38d4e4`
- Final context rank 1: `f2f20543-bb18-4405-9efa-32ddaef845a1`, section `3 Results`, page 10, blocks `b000111`, `b000112`
- Final context rank 2: `8825e5ff-1a29-416c-96fd-8a90b8b6ad27`, section `3 Results`, pages 10-11, blocks including `b000113`, `b000115`
- Corrected page map: `b000115` maps to page 11

## Decision

q024 single live QA now passes. A full 50-item rerun still requires an explicit separate authorization.
