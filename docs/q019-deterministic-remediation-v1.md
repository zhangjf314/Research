# q019 Deterministic Remediation v1

## Scope

This remediation addresses the q019 Full QA failure without calling the LLM and without rerunning Full QA.

- Question ID: `q019`
- Original failure: `citation_validation:page`
- Primary root cause: retrieval/context selection mismatch
- Secondary issue: ambiguous model-visible page range for multi-page chunks without `block_page_map`
- Reranker: disabled
- Deep Research: not executed
- Verification artifact: `data/evaluation/q019-deterministic-remediation-v1.json`

## Code changes

1. Added a paper-scoped experiment-design retrieval route.
   - Queries such as “How are the target paper's experiments designed and evaluated?” use a larger deterministic candidate pool.
   - Experiment-design context selection prioritizes evidence describing experimental variables and empirical setup.
   - References, appendix summaries, caveats, and limitations are deprioritized for this route.

2. Added deterministic `block_page_map` propagation for legacy chunks.
   - If a chunk already has a block-level page map, it is preserved.
   - If a legacy chunk has no map, every block in the chunk is explicitly mapped to `page_start`.
   - This makes validator behavior and model-visible allowed citations agree.

3. Disambiguated model-visible citation payload.
   - The LLM evidence payload now exposes only pages that appear in `allowed_citations`.
   - It no longer exposes the full `page_start..page_end` range when validation only allows page-specific triples.

## Production retrieval verification

The production container was rebuilt and recreated, then q019 retrieval was reproduced via:

```text
POST /api/v1/retrieve
query = How are the target paper's experiments designed and evaluated?
paper filter = 2001.08361 / d52a2824-279d-4f19-8118-a4d0ce423544
recall_k = 20
top_k = 10
```

Result:

- Status: `PASSED`
- Final context count: `3`
- Pre-rerank candidate count: `66`
- Reranker: disabled
- Gold chunk in final context: `true`

The q019 gold chunk is now the first final context item:

```text
chunk_id = 42164200-162e-47c7-9dbc-d7b4a4a98291
page_start = 7
page_end = 8
gold blocks = b000215, b000216, b000217, b000218
block_page_map = 7 for all four gold blocks
section = 3 Empirical Results and Basic Power Laws
```

## What this proves

This proves the deterministic retrieval/context input problem has been fixed for q019: page 7 gold evidence is now included in the production context before any LLM call.

It also removes the deterministic citation payload ambiguity where a multi-page chunk could display a page range while the strict validator accepted only page-start triples.

## What this does not prove

This does not prove q019 Full QA passes. No new q019 QA request was made, no LLM was called, and no Full QA rerun was executed in this remediation step.

Full QA remains `COMPLETED_WITH_FAILURES` until an explicitly authorized live rerun confirms q019 passes strict citation validation.

## Single live QA validation

After the deterministic remediation, an explicitly authorized q019 single live QA was executed with:

```text
scripts/run_production_qa_smoke_v1.py --sample-id q019 --single-attempt --no-json-repair --no-qa-retry
```

Result:

- Status: `PASSED`
- Real model called: `true`
- Provider/model: `siliconflow` / `Qwen/Qwen3-8B`
- Prompt: `qa-production-v1`
- Answerable: `true`
- Claim count: `4`
- Citation count: `4`
- Citation context validity: `1.0`
- Citation pages: `7, 7, 7, 5`
- Retrieved context count: `3`
- Gold chunk in context: `true`
- Input/output/total tokens: `9812 / 601 / 10413`
- QA endpoint latency: `13019.14 ms`
- Output artifact: `artifacts/q019-live-retry-result-v1.json`

This validates that the deterministic q019 fix can pass strict citation validation in a single live QA call.
The 50-item Full QA aggregate remains from the previous rerun until another explicitly authorized full rerun is executed.
