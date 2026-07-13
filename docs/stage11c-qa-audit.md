# Stage 11C: SiliconFlow Qwen QA audit

## Scope

Stage 11C changes only ordinary QA generation and evaluation. It fixes retrieval to
`jina-embeddings-v5-text-small` (1,024 dimensions), Structural Hybrid Recall 20 / Top
10, collection `papers_jina_eval34_v2__20260713152149`, and
`production-corpus-v1`. `RERANK_ENABLED=false`; Deep Research is not executed.

The target provider is SiliconFlow and the target model is `Qwen/Qwen3-8B`. The
adapter uses `POST https://api.siliconflow.cn/v1/chat/completions`, JSON mode,
`enable_thinking=false`, temperature 0, and prompt `qa-production-v1`.

## Pre-implementation audit

The repository already had a common `LLMProvider`, deterministic Template provider,
an OpenAI-compatible HTTP adapter, a QA route, basic model usage and latency fields,
and a lexical post-generation support filter. It did not have:

- a SiliconFlow-specific configuration and failure boundary;
- an exact answerable/refusal schema;
- claim IDs with paper/page/block citation triples;
- strict context-membership validation before accepting a response;
- bounded, recorded malformed JSON/schema/citation retries;
- rate-limit, API request, and retry diagnostics;
- token-budget context truncation and deduplication trace;
- a resumable smoke/dev/full QA evaluator over the signed retrieval protocol.

## Implemented offline behavior

Production configuration fails when provider, key, model, base URL, or the fixed
temperature is invalid. No path falls back from SiliconFlow to Template. HTTP errors
are sanitized and response bodies are not logged. Retry reasons use bounded categories
such as `malformed_json`, `schema_validation`, `citation_validation`, `HTTP 429`, and
timeout/network exception names.

An answerable result requires at least one atomic claim and every claim requires at
least one citation. Every citation triple must be present in the supplied context.
Unanswerable results require `answer=null`, no claims/citations, and a refusal reason.
First-token latency is `null` because the formal adapter uses non-streaming JSON mode.

Context is deduplicated in retrieval rank order and truncated using a configured token
budget. Each query records input candidates, retained chunk IDs, duplicates, estimated
tokens, the truncated chunk ID, and whether any gold block entered the final context.

## Rule-based evaluation

The evaluator uses approved gold answerability, required claims, paper IDs, block IDs,
pages, and retrieved-context membership. Required-claim coverage is token-set recall
with a fixed threshold of 0.35; every raw score is retained. Citation precision requires
the gold paper/page/block triple. No LLM judge or self-evaluation is used.

Modes are deterministic:

- smoke: `q001`, `q005`, `q030` (one answerable and both unanswerable records);
- dev: ten records (`q001`-`q009` plus `q030`);
- full: all 50 records.

`--resume` retains completed IDs and does not repeat them. Failures remain in the JSON
artifact and are eligible for a later retry. `--max-requests` is always available as a
safety limit; `--max-cost` is also supported when per-million-token rates are configured.

## Real-run audit

The real run completed on 2026-07-14. The initial connectivity call reached
SiliconFlow but exhausted three schema-validation attempts because the Prompt did not
show the exact JSON shape. Adding the explicit schema fixed connectivity: one request,
270 input tokens, 98 output tokens, 4.252 seconds, one claim, and one valid citation.

The first smoke command was rejected before retrieval because the inherited local
environment still had `RERANK_ENABLED=true`. Formal commands therefore used a
process-only `RERANK_ENABLED=false` override; no environment file was edited.

Smoke completed 3/3 with no retry. Its single answerable item did not retrieve a gold
block, so citation precision and recall were both zero even though schema and context-ID
validity were one. This distinction is preserved rather than calling smoke a quality
success.

The first Dev attempt retained two final failures: `q003` and `q007` each exhausted
three `citation_validation` attempts. Removing `chunk_id` from the model-visible
evidence fixed `q003`. `q007` continued to combine a valid block with a page from a
different context item; categorized retries recorded `citation_validation:page`.
Adding an explicit `block_page_map` removed that ambiguity without relaxing validation.
The completed Dev artifact contains 10/10 queries, one retained successful retry, and
no final failures.

Full completed 50/50 with two retained successful retries: `q003` retried after
`citation_validation`, and `q049` retried after `malformed_json`. There were no final
failures, fallbacks, or rate-limit events. Historical failed Dev attempts are recorded
in this audit because successful resume replaces the active row for that question.

## Final evidence

- Answerable accuracy: 0.875 (42/48); refusal accuracy: 1.0 (2/2).
- Required-claim coverage: 0.388889; unsupported claims: 147.
- Citation presence, ID validity, binding, and context membership: 1.0.
- Citation precision: 0.103009; citation recall: 0.096875.
- Gold-block retrieved rate among answerable questions: 0.4375.
- Total latency mean/p50/p95: 18.701 / 14.912 / 42.171 seconds.
- Input/output/total tokens: 307,953 / 20,753 / 328,706.
- API requests: 52; final retries: 2; rate-limit events: 0.
- First-token latency: unavailable (`null`) for non-streaming JSON mode.
- Cost: unavailable because token prices were not configured; it is not reported as
  zero cost.

The integration and evaluation workflow satisfy Stage 11C's engineering objective, but
the measured retrieval, claim coverage, and citation quality do not support a
production-quality QA claim. Stage 11D may proceed only as a bounded diagnosis and
improvement phase that keeps this baseline intact; the current QA path should not be
promoted as v1.0 quality.
