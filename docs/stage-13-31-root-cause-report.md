# Stage 13.31 Root Cause Report

## Outcome

The original q002 `/api/v1/qa` `HTTP 503` with `claim QA unavailable: ValueError`
was caused by a Docker runtime prompt-version mismatch.

- Failed request ID from Stage 13.30: `d6e306ef-11b5-46d4-929f-e724a93560f3`
- Failure stage: `LLM_REQUEST_BUILD`
- Root cause: container `PROMPT_VERSION=claim-qa-v1`
- Supported production prompt: `qa-production-v1`
- Precise code path: `SiliconFlowLLMProvider.generate_claim_answer()` called
  `qa_system_prompt(prompt_version)`, which raised `ValueError("unsupported production QA prompt version: claim-qa-v1")`.
- Why provider preflight passed: the preflight used a minimal completion path and did
  not render the production QA prompt, so it verified model reachability but not the
  Claim QA prompt contract.

## Fix

- `.env`, `.env.example`, and `docker-compose.yml` now align on `PROMPT_VERSION=qa-production-v1`.
- `Settings.prompt_version` default is `qa-production-v1`.
- `QAService`/`Answer` defaults were aligned to `qa-production-v1`.
- Prompt mismatch is now returned as a structured 503 with:
  - `code=CLAIM_QA_CONFIGURATION_ERROR`
  - `stage=LLM_REQUEST_BUILD`
- Provider parse/schema/citation failures now carry stable error code and stage metadata.
- No Template fallback was introduced.

## Post-fix smoke result

The single authorized q002 smoke reached the real provider:

- Provider/model: `siliconflow` / `Qwen/Qwen3-8B`
- Prompt version: `qa-production-v1`
- Template fallback: `false`
- Token usage: input `12926`, output `43`, total `12969`
- API request status: real model returned an answer envelope
- Smoke status: `FAILED`
- Current blocking class: `BLOCKED_BY_CONTEXT_DATA`

The failure changed from an internal configuration error to a retrieval/context issue:
q002 is answerable, but the retrieved context contained reference/visualization blocks and
did not include the gold evidence (`page=2`, `block_id=b000025`).

## Retrieval/context follow-up

The q002 context issue was reproduced without additional LLM calls. The target chunk
existed in production data:

- Paper UUID: `2537b3aa-d6aa-4a4a-aac1-477fc58bc3d9`
- Chunk: `35be87a6-5aec-4267-bb68-42e82bcd0235`
- Page: `2`
- Gold block: `b000025`

Before the context fix, that chunk was recalled but ranked too low for the QA context:

- Dense rank: `15`
- Sparse rank: `15`
- Fusion rank: `18`
- Context sent to q002: References / Attention Visualizations only

A generic, non-oracle context-ordering fix was added for paper-scoped contribution
questions. It reorders the already-retrieved candidate set before context construction,
prioritizing Introduction / Abstract-like / Conclusion sections and de-prioritizing
References / visualizations. After the fix, q002 `/retrieve` includes the gold chunk as
context rank `1`.

A second q002 live LLM smoke was run only after explicit user authorization. It passed:

- Provider/model: `siliconflow` / `Qwen/Qwen3-8B`
- Prompt version: `qa-production-v1`
- Claims/citations: `4` / `4`
- Citation context validity: `1.0`
- Tokens input/output/total: `4213` / `474` / `4687`
- Template fallback: `false`

## Gate status

- `LIVE_QA_SMOKE_GATE=PASSED`
- `Q002_RETRIEVAL_CONTEXT_GATE=PASSED`
- `PRODUCTION_FULL_QA_GATE=READY_TO_RUN`
- Full QA: not run
- Deep Research: not run
