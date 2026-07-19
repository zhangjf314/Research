# Retrieval Generalization Audit v1

- Engineering gate: `PASSED`
- Dataset policy: `portfolio-evaluation-policy-v1`
- gold-dev-v1: `50` samples, `50` approved; internal development evaluation set, not blind holdout
- retrieval-diagnostic-v1: `27` claim-level samples; diagnostic, not blind
- Sample size sufficient: `False`
- Dev retrieval gate: `PASSED`
- Diagnostic gate: `PASSED`
- Validation gate: `DIAGNOSTIC_NOT_HOLDOUT`
- Holdout gate: `NOT_EVALUATED`
- Shadow pilot gate: `NOT_EVALUATED`
- Generalization gate: `DIAGNOSTIC_ONLY`
- Generalization evidence: `DIAGNOSTIC_ONLY`
- Strong generalization claim allowed: `False`
- Holdout contaminated: `True`
- Shadow pilot required for Full QA: `False`
- NEXT_LIVE_READY: `True`
- READY_FOR_FULL_QA: `True`

## Limitations

- gold-dev-v1 is an internal development evaluation set, not a blind holdout
- retrieval-diagnostic-v1 has been used during development
- shadow-holdout-pilot-v1 has not been created
- large-scale independent blind benchmark evidence is unavailable

## Portfolio-safe claims

- 基于 50 条人工审核的内部评测数据完成检索和问答评测
- 27 条 claim-level diagnostic 集用于失败分析和回归检查

## Forbidden claims

- strict generalization benchmark
- large-scale independent blind benchmark
- production-grade generalization proven
- statistically sufficient holdout

Full QA is no longer blocked solely by the absence of a 50-sample strict blind shadow holdout. The project may proceed to Production Full QA under the portfolio policy while explicitly disclosing that large-scale independent blind generalization evidence is not available.
