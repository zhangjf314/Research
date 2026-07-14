# Stage 11C.7 Citation Audit Decision

## Scope and provenance

This is an **AI-assisted manual citation audit, 30-sample stratified review**. It is not
an independent human double-blind audit and does not confirm citation precision on the
full QA dataset. The sample was deliberately enriched for suspected failures: ten
semantic non-Gold citations, ten same-page non-exact citations, and ten weak/unsupported
citations.

All 30 records are approved, have unique IDs, complete reviewer metadata, and no pending
labels. The immutable claim, citation, Gold evidence, and automated-label fields match the
Stage 11C.6 source records.

## Human-calibrated citation quality

- Fully supported: 5/30 (16.7%).
- Fully or partially supported: 7/30 (23.3%).
- Related but insufficient: 7/30 (23.3%).
- Unsupported: 16/30 (53.3%).
- Gold annotation too narrow: 0/30.
- Token-set semantic strict/lenient precision: 30%/40%.
- Same-page citations judged unsupported: 6/10.
- Weak citations judged unsupported: 6/8.
- Automated unsupported negative precision: 2/2 (100%).

The Stage 11C.5 retrieved-mode 81.6% semantic-support figure is a token-overlap signal,
not citation correctness. The stratified review demonstrates material overestimation of
human-confirmed support. Historical Stage 11C.5 and 11C.6 JSON/CSV artifacts remain
unchanged; this document supplies the corrected interpretation.

## q033 and q044 protocol ruling

The old protocol serialized every block in a multi-page chunk with `page_start`, while
the validator accepted a broader chunk-page/block Cartesian product. Citation retries
then resent the same request without the failed mapping or an authoritative legal-triple
list. Invalid model bodies were not retained historically, so they are explicitly marked
`NOT_RETAINED_BY_STAGE_11C6` rather than reconstructed.

The repair adds exact `block_page_map` support, an `allowed_citations` triple list, strict
map validation, sanitized diagnostic attempts, and a bounded correction prompt listing
legal triples. It never changes a model-generated page and never relaxes block/page
validation.

Minimal replay used only q033 and q044 with the frozen Qwen3-8B model and prompt version:

| Question | First attempt | Retry | Requests | Result |
|---|---|---|---:|---|
| q033 | Illegal page, strictly rejected | Legal triples followed | 2 | Completed |
| q044 | Illegal page, strictly rejected | Legal triples followed | 2 | Completed |

Both failures are ruled deterministic protocol/retry defects. Because both recovered
after one explicit bounded retry, they are not classified as a persistent
`provider_model_citation_mapping_limitation`. The first invalid outputs remain recorded
as rejected evidence in the diagnostic artifacts.

## Stage 11D decision

Full quality evaluation remains blocked: the reviewed sample has only 16.7% strict and
23.3% lenient support. A maximum of three **engineering-only bounded smoke** cases may
proceed because the audit sample is failure-enriched and both deterministic citation
failures are fixed. This permission is conditional on all of the following hard limits:

- exactly 3 smoke cases; no expansion to a benchmark run;
- at most 2 research iterations per case;
- at most 4 LLM requests per case and 12 total;
- at most 40,000 tokens per case and 120,000 total;
- configured model pricing and a USD 1.00 total stop cap before execution;
- Reranker disabled and the existing Embedding, LLM, prompt, corpus, and Gold frozen;
- strict citation validation enabled, with no automatic page correction;
- every result labelled engineering-only and excluded from v1.0 quality evidence.

If model pricing is not configured, the smoke remains operationally blocked because the
cost stop condition cannot be enforced. Deep Research quality acceptance and any
production-ready or v1.0 claim remain prohibited.
