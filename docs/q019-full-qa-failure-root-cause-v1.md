# q019 Full QA Failure Freeze and Root-Cause Analysis v1

## Scope

This document freezes the q019 failure observed in the authorized 50-item Production Full QA rerun after the Stage 13.32 context-selection change.

- Question ID: `q019`
- Dataset: `gold-dev-v1`
- Full QA status: `COMPLETED_WITH_FAILURES`
- Failed item: `q019`
- Provider/model: SiliconFlow `Qwen/Qwen3-8B`
- Prompt: `qa-production-v1`
- Reranker: disabled
- Deep Research: not executed
- Freeze JSON: `data/evaluation/q019-full-qa-failure-freeze-v1.json`

## Terminal failure

q019 failed in local strict citation validation:

- Provider error code: `CLAIM_QA_CITATION_VALIDATION_ERROR`
- Stage: `CLAIM_CITATION_VALIDATE`
- Reason: `citation_validation:page`
- API request count: `1`
- Request ID: `03a7e352-e7e2-40e9-a04c-8931c7f2dd3f`
- Rate-limit events: `0`

The provider response audit shows:

- HTTP status: `200`
- Finish reason: `stop`
- Content present: `true`
- JSON parse error: `null`
- Usage reported: `11655 / 543 / 12198`
- Full raw payload persisted: `false`
- Audit file: `artifacts/private/qa-response-audits/q019-full-qa-rerun-q019-1784391762.json`

Therefore this is not a model JSON parse failure and not an API response wrapper failure.

## Retrieval/context reproduction

The Full QA query was reproduced through `/api/v1/retrieve` without calling `/qa` or the LLM:

```text
query = How are the target paper's experiments designed and evaluated?
paper filter = 2001.08361 / d52a2824-279d-4f19-8118-a4d0ce423544
Full QA recall_k = 20
Full QA top_k = 10
```

The gold evidence for q019 is:

- `b000215`
- `b000216`
- `b000217`
- `b000218`
- gold page: `7`

In the production Jina chunk file, those gold blocks are in chunk:

```text
chunk_id = 42164200-162e-47c7-9dbc-d7b4a4a98291
page_start = 7
page_end = 8
block_page_map = null
```

That gold chunk was not in the Full QA final context. With a diagnostic `recall_k=100`, it was still only:

- dense rank: `45`
- sparse rank: `41`
- fusion rank: `47`
- final context: not selected

The actual reproduced final context contained four non-gold chunks:

| Rank | Chunk | Pages | Section | Contains q019 gold |
|---:|---|---:|---|---|
| 1 | `7c070517-75ed-4580-9c55-17e84da72b5c` | 29 | References | false |
| 2 | `a4463f0a-ec46-41d1-bcca-c256e0aea15e` | 20 | Appendix summary of power laws | false |
| 3 | `38277725-1b05-4e52-8e40-c4e34b9ce263` | 14 | Scaling laws with model size and training time | false |
| 4 | `92de0fb3-3f90-4d9b-8823-e8a2aafb7d1d` | 15-16 | Optimal allocation of compute budget | false |

## Root cause

Primary root cause:

```text
RETRIEVAL_CONTEXT_SELECTION_MISMATCH
```

The q019 gold evidence is about the experiment design variables on page 7, but the production retrieval route selected later sections, appendix summaries, and references. The gold chunk was ranked too low to enter Full QA recall/context selection.

Secondary terminal failure:

```text
MODEL_CITED_EXISTING_CONTEXT_BLOCK_WITH_INVALID_PAGE_OR_CONTEXT_PAGE_MAP_AMBIGUITY
```

The validator returned `citation_validation:page`, which means the cited `paper_id` and `block_id` existed in supplied context, but the cited page did not match the allowed triple.

One reproduced context chunk spans pages `15-16` and has `block_page_map=null`. In this case the validator falls back to `page_start` for all block IDs, while the model-visible payload can still expose a page range. This creates a deterministic ambiguity for multi-page chunks. The exact hidden invalid citation cannot be reconstructed because full payload persistence was disabled and the audit captured only prefix/suffix windows.

## Classification

- Citation selection problem: yes, but downstream of wrong/irrelevant context.
- Context mapping problem: yes, for multi-page chunks with missing `block_page_map`.
- Model page citation error: yes at the terminal validator level, because it cited a context block with a page outside the allowed triple.
- JSON/schema failure: no.
- API wrapper failure: no.
- Gold mutation issue: no.

## Recommended next step

Do not rerun q019 immediately. A deterministic remediation should first address both:

1. q019 retrieval/query routing and context selection for experiment-design questions.
2. block-level page mapping or model-visible citation payload ambiguity for multi-page chunks.

If another live rerun is authorized later, enable full sanitized payload persistence for failing samples only, so exact invalid citations can be frozen without exposing secrets.

