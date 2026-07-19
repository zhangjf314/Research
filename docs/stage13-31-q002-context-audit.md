# Stage 13.31 q002 Context Audit

- Sample: `q002`
- Retrieval scope: `paper`
- Gold paper IDs: `['1706.03762']`
- Gold pages: `[2]`
- Gold block IDs: `['b000025']`
- Retrieved context count: `3`
- Stored evidence: metadata, hashes, lengths, and short snippets only.
- Fix status: `PASSED_RETRIEVAL_CONTEXT_SMOKE`
- Root cause before fix: the gold chunk existed and was recalled at fusion rank `18`, but
  generic paper-scoped retrieval sent the first Top-K candidates to the context builder.
  References and attention-visualization chunks ranked above the Introduction chunk and
  consumed the QA context budget before gold block `b000025` could be included.
- Fix applied: paper-scoped contribution questions now use section-aware context ordering
  over the already-retrieved candidate set. This prioritizes Introduction / Abstract-like /
  Conclusion chunks and de-prioritizes References / visualization sections for the context
  sent to QA. It does not rewrite the query, inspect Gold, inject oracle evidence, modify
  Qdrant, enable reranker, or call an LLM.
- Post-fix result: q002 context rank `1` is chunk
  `35be87a6-5aec-4267-bb68-42e82bcd0235`, page `2`, containing gold block `b000025`.

| rank | page | score | block ids | text sha256 | length |
|---:|---:|---:|---|---|---:|
| 1 | 2 | 0.02666666666666667 | `b000022,b000023,b000024,b000025` | `54dc176d2202bf4c` | 5942 |
| 2 | 1 | 0.027809742999616416 | `b000017,b000018,b000019,b000020` | `ac31b79c91e246f8` | 4516 |
| 3 | 10 | 0.029418126757516764 | `b000164,b000165,b000166,b000167,b000168` | `42b3d8096336a5ea` | 1542 |
