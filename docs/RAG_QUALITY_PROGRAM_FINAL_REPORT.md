# RAG Quality Optimization Program Final Report

Status: `RAG_QUALITY_OPTIMIZATION_PROGRAM_CLOSED`

The RAG Quality v3-v5 program is closed. Candidate generation is strong under
a validated Gold-free runtime/evaluation boundary, but no post-retrieval policy
met frozen requirements for average quality, fresh-blind generalization, and
tail safety together. The selected candidate is `NONE`; Full QA was not run and
production P0 remains unchanged.

## Consolidated evidence

| Program | Measured upside | Frozen rejection reason |
| --- | --- | --- |
| V3 C2 Qwen reranker | GoldR@5 `.603084 -> .723634`; Claim@5 `.763158 -> .907895`; ranking loss `19 -> 7`. | Fresh B1 blind generalization failed. |
| V3 D2B listwise selector | GoldR@5 `.738812 -> .849738`; Claim@5 `.852941 -> .955882`; MultiComplete@5 `.562500 -> .875000`; ranking loss `24 -> 4`. | 9 new DEV tail regressions; maximum was 2. |
| V4 B2 variable-K selector | Context precision `.228378 -> .542793`; non-Gold context `3.858108 -> .878379`. | Required evidence and tails regressed. |
| V5 A verifier/controller | SUPPORT recall `.971751` on the sealed evaluation. | The controller did not establish safe evidence sufficiency. |

The bounded scientific conclusion is that fixed Top-5 over-selects context,
while the tested aggressive variable-K rule under-selects required evidence.
Pointwise and listwise methods showed average value but did not establish robust
promotion evidence. The supervised sufficiency signal was not reliable enough
for a safe stop or selection policy. No experimental reranker, selector, or
verifier is deployed.

Historical A1D/A2B quality evidence is retained for audit but invalidated by
Gold-dependent attribution. FINAL_BLIND_V4 and FINAL_BLIND_V5 were not
consumed. The next stage is not planned.

See the [program evidence ledger](rag-quality-program-evidence-ledger.md).
