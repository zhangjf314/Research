# RAG Quality Optimization Program Evidence Ledger

Status: `RAG_QUALITY_OPTIMIZATION_PROGRAM_CLOSED`

This sanitized ledger indexes the closed RAG Quality v3-v5 program. It keeps
the valid measurements, rejected promotion decisions, and invalidated
historical evidence while excluding evaluation corpora, raw provider traces,
candidate snapshots, local indexes, credentials, and private operations.

## Final program classifications

- `RAG_QUALITY_OPTIMIZATION_PROGRAM_CLOSED`
- `RAG_QUALITY_V3_CLOSED`
- `RAG_QUALITY_V4_CLOSED`
- `RAG_QUALITY_V5_CLOSED`
- `CANDIDATE_GENERATION_STRONG`
- `NO_ROBUST_POST_RETRIEVAL_POLICY_VALIDATED`
- `NO_CANDIDATE_PROMOTED`
- `PRODUCTION_P0_RETAINED`
- `FINAL_BLIND_V4_V5_NOT_CONSUMED`
- `FULL_QA_NOT_RUN`
- `PORTFOLIO_RAG_EVIDENCE_COMPLETE`
- `NO_FURTHER_RAG_QUALITY_STAGE_PLANNED`

## Evidence register

| Evidence | Status | Bounded result |
| --- | --- | --- |
| V3 A1D/A2B historical quality evidence | INVALIDATED | Gold-dependent attribution invalidated its quality/promotion use; records are retained for audit only. |
| V3 C1 Gold-free baseline | VALIDATED | GoldR@5 `.603084`, MRR `.577841`, NDCG@10 `.532461`, context precision `.209091`, Claim@5 `.763158`; losses `5/19/0` identify ranking as primary. |
| V3 C2 pointwise reranker | VALIDATED, development only | GoldR@5 `.603084 -> .723634`, Claim@5 `.763158 -> .907895`, ranking loss `19 -> 7`. |
| V3 B1 blind reranker | REJECTED | Average metrics improved, but frozen blind generalization failed: comparison Gold regression `-.166666`, MRR CI `[-.052500, .192229]`, NDCG CI `[-.023752, .178728]`, top-two share `.562500 > .5`. |
| V3 D2B listwise selector | REJECTED | Strong mean gains, but 9 new DEV GoldR tail regressions exceeded the frozen maximum of 2. |
| V4 B2 facet-aware variable-K | REJECTED | Context precision `.228378 -> .542793`, but GoldR `.805100 -> .597410`, Claim `.926020 -> .729592`, MultiComplete `.209184 -> .153061`; tail safety failed. |
| V5 A verifier/controller | REJECTED | Macro F1 `.439898`, NEUTRAL F1 `0`, false-sufficient risk `.565217`, false-insufficient `.807692`, recall `.192308`, F1 `.266667`. |
| FINAL_BLIND_V4/V5 and Full QA | NOT_RUN | Neither final blind protocol was consumed and Full QA was not run. |

## V5 false-sufficient metric semantics audit

The frozen gate value `.565217` is false-positive insufficient sets divided by
sets predicted `SUFFICIENT`. The historical paper-bootstrap interval
`[0, .139957]` is instead a per-paper false-sufficient incident rate with all
evaluated sets as its denominator. The shared historical label is a
**metric-reporting ambiguity**, not a recalculation, frozen-gate, or
classification error. The frozen artifact remains unchanged.

## Closure boundary

This closure made zero provider, retrieval, embedding, reranker, LLM,
verifier-training, blind-evaluation, or production calls.
