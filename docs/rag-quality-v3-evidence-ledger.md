# RAG Quality v3 — Final Evidence Ledger

Status date: final closure. This ledger is a mechanical index of the preserved stage artifacts; it does not authorize a new experiment, candidate, blind set, Full QA run, or production change.

## Final program state

- RAG_QUALITY_V3_CLOSED
- POST_RETRIEVAL_OPTIMIZATION_CLOSED
- NO_CANDIDATE_PROMOTED
- PRODUCTION_P0_RETAINED
- FULL_QA_NOT_ELIGIBLE
- PORTFOLIO_EVIDENCE_READY

## Ledger

| Stage | Status | Preserved conclusion | Authoritative artifact |
|---|---|---|---|
| C0 | VALIDATED | Gold-free runtime and attribution boundary were implemented and tested. Historical A1D/A2B development evidence is not salvageable. | artifacts/rag-quality-v3/c0/final/c0-final-decision-v1.json |
| C1 | REJECTED | Fresh Gold-free, isolated development execution completed. Representation alternatives were not sufficient; PRIMARY_LOSS=ranking_loss. No representation was selected. | artifacts/rag-quality-v3/c1/execution/final/c1-final-decision-v1.json |
| C2 | VALIDATED | Clean pointwise reranking passed the frozen development gate (C2-R1). This is development-only evidence. | artifacts/rag-quality-v3/c2/execution/final/c2-final-decision-v1.json |
| B1 | REJECTED | Fresh blind generalization of the C2 reranker failed; it does not make Full QA eligible. The consumed B1 evidence is no longer blind for later development stages. | artifacts/rag-quality-v3/b1/execution/final/b1-final-decision-v1.json |
| D0 | VALIDATED | Root cause was MIXED_RANKING_FAILURE: 8 pointwise, 5 cross-section, and 4 set-completeness residual losses. | artifacts/rag-quality-v3/d0/d0-public-summary-v1.json |
| D1 | REJECTED | Static RRF fusion and the fixed section-aware heuristic were insufficient; no candidate was selected. | artifacts/rag-quality-v3/d1/execution/d1-final-decision-v1.json |
| D2 | NOT_RUN | Selector quality is not evaluated: 236/236 selector calls ended with finish_reason=length, valid outputs were zero, and fail-closed R1 equaled R0. | artifacts/rag-quality-v3/d2/execution/d2-final-decision-v1.json |
| D2B | REJECTED | Valid listwise execution completed (236/236 valid outputs) and improved averages, but the frozen DEV tail-safety gate failed (9 new GoldR regressions; maximum 2). No candidate was promoted. | artifacts/rag-quality-v3/d2b/execution/d2b-final-decision-v1.json |

## Invalidated historical evidence

The following artifacts remain retained for traceability, but are **INVALIDATED** as quality or promotion evidence because Gold-dependent attribution was found in their historical development paths:

| Historical evidence | Status | Reason | Preservation record |
|---|---|---|---|
| A1D quality / promotion evidence | INVALIDATED | Gold-dependent attribution; C0 found clean-rescore coverage 0/17558. | artifacts/rag-quality-v3/c0/final/c0-final-decision-v1.json |
| A2B quality / promotion evidence | INVALIDATED | Gold-dependent attribution; C0 found clean-rescore coverage 0/3520. | artifacts/rag-quality-v3/c0/final/c0-final-decision-v1.json |

Invalidated does not mean deleted: the historical artifacts remain in the repository and must not be used to justify promotion.

## Not run

| Item | Status | Boundary |
|---|---|---|
| B2 fresh blind | NOT_RUN | No D-stage candidate passed its frozen gate. |
| Full QA | NOT_RUN | Not eligible after B1/D-stage outcomes. |
| Production promotion/change | NOT_RUN | Production P0 is retained. |
| D3/D4 or further selector search | NOT_RUN | Explicitly prohibited by final closure. |
