# RAG Quality v3 — Portfolio Evidence Summary

## Portfolio-facing status

RAG Quality v3 is closed with a defensible negative promotion decision:

- Gold-free runtime work is validated.
- No representation candidate was promoted.
- Pointwise reranking showed clean development benefit but failed fresh blind generalization.
- Post-retrieval listwise selection produced valid, strong average development-visible gains, but failed its precommitted tail-safety condition.
- Production remains P0; Full QA is not eligible.

This document is public-facing and intentionally contains no credentials, local paths, private operational notes, or personal materials. It summarizes evidence boundaries; it does not claim a deployed quality improvement.

## Evidence classification

| Classification | Evidence |
|---|---|
| VALIDATED | Gold-free runtime boundary (C0/C1); C2 pointwise development result; D0 ranking-failure diagnosis; D2B selector execution validity and its measured averages. |
| REJECTED | C1 representation promotion; B1 blind generalization; D1 static hybrid/heuristic selection; D2B promotion under the frozen tail-safety gate. |
| INVALIDATED | A1D and A2B historical quality/promotion evidence, due to Gold-dependent attribution. |
| NOT_RUN | B2 fresh blind, Full QA, production promotion, and any D3/D4 successor. |

## Final decision record

RAG_QUALITY_V3_CLOSED · POST_RETRIEVAL_OPTIMIZATION_CLOSED · NO_CANDIDATE_PROMOTED · PRODUCTION_P0_RETAINED · FULL_QA_NOT_ELIGIBLE · PORTFOLIO_EVIDENCE_READY

Consult the final report and evidence ledger for the immutable evidence boundary and source artifacts.
