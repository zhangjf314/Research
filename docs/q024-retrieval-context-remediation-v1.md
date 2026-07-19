# q024 Retrieval/Context Remediation v1

Status: implemented and retrieval-verified.

This remediation is deterministic. It does not run live QA, does not run Full QA, does not enable reranking, does not repair citations, and preserves strict paper/page/block citation validation.

## Root cause

The q024 failure was a retrieval/context input problem rather than an API wrapper or JSON schema problem. The production path did not reliably present the GPT-3 Results/evaluation evidence needed for the experiment-design question. A second issue was stale dense/Qdrant chunk metadata for multi-page chunks: the chunk spanning pages 10-11 could carry a `block_page_map` that mapped all blocks to page 10.

## Deterministic fix

- Paper-scoped experiment-design queries now receive auditable routing signals:
  - models compared
  - task categories
  - scaling curves
  - evaluated on datasets
  - evaluate the 8 models
  - wide range of datasets
  - group the datasets
  - training curves
  - power-law trend
- Retrieval trace now records both the original query and the routed query/signals.
- Experiment-design context selection prioritizes Results/evaluation chunks containing those signals over generic Approach/Introduction context.
- RRF still preserves dense/sparse ranks and scores, but for duplicate chunk IDs it keeps the sparse/BM25 chunk payload as the fresh parsed-corpus metadata source.
- API chunk loading backfills or corrects `block_page_map` from `paper_blocks.jsonl`.

## q024 retrieval verification

Command class: `POST http://localhost/api/v1/retrieve`

- Query: `How are the target paper's experiments designed and evaluated?`
- Paper filter UUID: `930bea6f-5263-4012-8451-c2d19c38d4e4`
- Paper ID: `2005.14165`
- Retrieval scope: `paper`
- Reranker API requests: `0`

Final context included the two GPT-3 Results gold chunks:

| Rank | Chunk | Pages | Gold blocks |
| --- | --- | --- | --- |
| 1 | `f2f20543-bb18-4405-9efa-32ddaef845a1` | 10-10 | `b000111`, `b000112` |
| 2 | `8825e5ff-1a29-416c-96fd-8a90b8b6ad27` | 10-11 | `b000113`, `b000115` |

The corrected per-block page map for the multi-page chunk is:

- `b000113`: page 10
- `b000114`: page 10
- `b000115`: page 11
- `b000116`: page 11
- `b000117`: page 11
- `b000118`: page 11
- `b000119`: page 11

## q019 regression check

The same paper-scoped experiment-design retrieval path still returns the q019 gold chunk in final context:

- Paper filter UUID: `d52a2824-279d-4f19-8118-a4d0ce423544`
- Paper ID: `2001.08361`
- Gold blocks present: `b000215`, `b000216`, `b000217`, `b000218`
- Gold block pages: page 7

## Remaining state

The last Full QA run remains `COMPLETED_WITH_FAILURES` because q024 has not been rerun after this remediation. A q024 single live QA or a new 50-item Full QA rerun requires explicit authorization.
