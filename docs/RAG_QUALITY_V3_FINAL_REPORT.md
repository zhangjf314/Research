# RAG Quality v3 — Final Closure Report

## Decision

RAG_QUALITY_V3_CLOSED
POST_RETRIEVAL_OPTIMIZATION_CLOSED
NO_CANDIDATE_PROMOTED
PRODUCTION_P0_RETAINED
FULL_QA_NOT_ELIGIBLE
PORTFOLIO_EVIDENCE_READY

The final selected candidate is NONE. B2 fresh blind eligibility is no. Full QA was not run and production was not changed.

## Validated findings

- Gold-free runtime and evaluation attribution were validated by C0/C1. Gold was removed from new runtime payloads, and fresh C1 execution used Gold-free isolated indexes.
- Candidate generation was comparatively strong in the fresh C1 evidence. C1 identified ranking loss as the primary remaining loss, while all frozen representation candidates failed the promotion gate.
- C2 validated clean pointwise reranking on development evidence only.
- D0 validated MIXED_RANKING_FAILURE as the residual taxonomy: 8 pointwise, 5 cross-section, and 4 set-completeness cases.

## Rejected findings and interventions

- C1 rejected representation optimization as sufficient evidence for promotion.
- B1 rejected fresh blind generalization of the C2 pointwise reranker.
- D1 rejected static RRF fusion and the fixed section-aware selector.
- D2B rejected listwise promotion despite valid execution and strong average gains, because its frozen tail-safety condition failed.

## D2 and D2B interpretation

D2 remains historically closed as POST_RETRIEVAL_SELECTION_PROGRAM_CLOSED_NO_PROMOTION, but its selector quality is NOT_EVALUATED: all 236 selector calls ended at the 128-token output limit, no valid selections were produced, and fail-closed selection made R1 identical to R0.

D2B changed only the prevalidated output budget from 128 to 256 tokens. It completed 236/236 valid strict-schema selections with no retrieval, embedding, or reranker calls.

| D2B combined metric | R0 | R1 |
|---|---:|---:|
| GoldR@5 | .738812 | .849738 |
| Claim@5 | .852941 | .955882 |
| MultiComplete@5 | .562500 | .875000 |
| Ranking loss | 24 | 4 |

Frozen residual recovery was 8/8 pointwise, 4/5 cross-section, and 4/4 set-completeness cases. However, DEV176 had 9 new GoldR tail regressions while the frozen maximum was 2. The frozen gate therefore failed and promotion was rejected. This preserves the average improvements as valid measurement evidence without treating them as a production recommendation.

## Invalidated historical evidence

A1D and A2B quality/promotion evidence is **INVALIDATED** by Gold-dependent attribution. The records are retained, not deleted, but are excluded from the final evidence basis. C0 records clean-rescore coverage of 0/17558 for A1D and 0/3520 for A2B.

## Closure boundaries

- Provider calls during final closure: 0
- New experiments: 0
- New candidates, blind sets, Full QA, and production promotion: prohibited and not run
- Production state: unchanged P0

The detailed status and artifact paths are maintained in the final evidence ledger.
