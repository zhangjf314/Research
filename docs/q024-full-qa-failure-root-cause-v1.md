# q024 Full QA Failure Freeze and Root-Cause Analysis v1

## Scope

This document freezes the q024 failure observed in the 50-item Production Full QA rerun after q019 was remediated.

- Question ID: `q024`
- Dataset: `gold-dev-v1`
- Full QA status: `COMPLETED_WITH_FAILURES`
- Failed item: `q024`
- Provider/model: SiliconFlow `Qwen/Qwen3-8B`
- Prompt: `qa-production-v1`
- Reranker: disabled
- Deep Research: not executed
- Freeze JSON: `data/evaluation/q024-full-qa-failure-freeze-v1.json`

## Terminal failure

q024 failed in local strict citation validation:

- Provider error code: `CLAIM_QA_CITATION_VALIDATION_ERROR`
- Stage: `CLAIM_CITATION_VALIDATE`
- Reason: `citation_validation:page`
- API request count: `1`
- Request ID: `fff74853-e656-43a2-a2b6-7c8baa28934c`
- Rate-limit events: `0`

The provider response audit shows:

- HTTP status: `200`
- Finish reason: `stop`
- Content present: `true`
- JSON parse error: `null`
- Usage reported: `4111 / 525 / 4636`
- Full raw payload persisted: `false`
- Audit file: `artifacts/private/qa-response-audits/q024-full-qa-rerun-q024-1784391885.json`

Therefore this is not a model JSON parse failure and not an API response wrapper failure.

## Gold evidence

The q024 gold evidence is:

- Paper: `2005.14165`
- Gold pages: `10, 11`
- Gold blocks: `b000111`, `b000112`, `b000113`, `b000115`

In the production Jina chunk file:

| Chunk | Blocks | Pages | Section |
|---|---|---:|---|
| `f2f20543-bb18-4405-9efa-32ddaef845a1` | `b000111`, `b000112` | 10 | `3 Results` |
| `8825e5ff-1a29-416c-96fd-8a90b8b6ad27` | `b000113`, `b000115` | 10-11 | `3 Results` |

## Retrieval/context reproduction

The q024 retrieval path was reproduced through `/api/v1/retrieve` without calling `/qa` or the LLM:

```text
query = How are the target paper's experiments designed and evaluated?
paper filter = 2005.14165 / 930bea6f-5263-4012-8451-c2d19c38d4e4
diagnostic recall_k = 100
top_k = 20
```

Target ranks:

| Target | Dense | Sparse | Fusion | Final context |
|---|---:|---:|---:|---|
| Gold `b000111/b000112` chunk | not in top 100 | not in top 100 | not present | absent |
| Gold `b000113/b000115` chunk | 27 | 16 | 10 | absent |
| Contributions `b000470/b000471` chunk | 3 | 12 | 3 | absent in current reproduction |

The current reproduced final context contains three non-gold Approach chunks:

| Rank | Chunk | Pages | Section | Contains q024 gold |
|---:|---|---:|---|---|
| 1 | `71e420d5-d8a6-4e2a-923e-6e3474b38845` | 7 | `2 Approach` | false |
| 2 | `6f59e068-f2f2-4fd0-9276-60d6961fb30a` | 9-10 | `2 Approach` | false |
| 3 | `a59eb8ae-18dd-47df-b4ae-265d1069c3fc` | 10 | `2 Approach` | false |

Each current context item has a deterministic single-page `block_page_map` for its blocks. No multi-page citation payload ambiguity like the earlier q019 case was observed in the reproduced q024 context.

## Model response evidence

The private audit suffix shows the model citing page-42 `Contributions` blocks:

- `b000470`
- `b000471`

Those blocks are not q024 gold evidence. Because full raw payload persistence was disabled, the exact first invalid citation that triggered `citation_validation:page` cannot be reconstructed from the prefix/suffix audit windows alone.

## Root cause

Primary root cause:

```text
RETRIEVAL_CONTEXT_SELECTION_MISMATCH
```

The q024 gold evidence lives in the `3 Results` section, but the current production route overselects `2 Approach` chunks. One Results gold chunk is retrievable at fusion rank 10 under a larger diagnostic candidate pool but is not selected into final context. The other Results gold chunk is not found in dense/sparse top 100.

Secondary terminal failure:

```text
MODEL_CITED_CONTEXT_BLOCK_WITH_INVALID_PAGE
```

The terminal API failure is still strict `citation_validation:page`, but unlike q019 this is not best explained by multi-page payload ambiguity. The reproduced context has explicit block-page maps, and the model appears to have generated at least one citation page that did not match the allowed triples.

## Classification

- Same broad failure family as q019: yes, both are strict citation page validation failures after a non-gold context path.
- Same deterministic cause as q019: no.
- Citation selection problem: yes.
- Retrieval/context problem: yes, primary.
- Context page-map ambiguity: not observed in current reproduction.
- Model page citation error: yes at terminal validator level.
- JSON/schema failure: no.
- API wrapper failure: no.
- Gold mutation issue: no.

## Recommended next step

Do not directly rerun q024. First apply a deterministic retrieval/context remediation for GPT-3 experiment-design questions:

1. Prioritize `Results`/evaluation sections over generic `Approach` chunks for experiment-design queries.
2. Add query normalization terms that capture “models compared”, “task categories”, “scaling curves”, and “evaluated on datasets”.
3. Preserve strict citation validation; do not auto-repair invalid pages.
4. If another live rerun is authorized, persist the full sanitized payload for failing samples only.

