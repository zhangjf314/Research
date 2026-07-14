# v1 Gap Closure Plan

This plan does not add features or promise dates. It orders evidence-backed work and requires
separate authorization before any real model request. Structured details are in
[`data/evaluation/v1-gap-closure-plan.json`](../data/evaluation/v1-gap-closure-plan.json).

## P0 — quality prerequisites

1. Improve claim-level citation support; require strict >=0.80, lenient >=0.90 and unsupported
   claim rate <=0.10 on representative human review.
2. Raise exact gold block availability from 0.416667 to >=0.80 and page availability from
   0.645833 to >=0.90 under the frozen, non-Oracle protocol.
3. Raise required claim coverage from 0.388889 to >=0.80 without citation regression.
4. Freeze a Deep Research quality manifest only after the preceding gates pass.

## P1 — reliability and calibration

1. Establish Provider health/failure evidence without silent fallback.
2. Pin compatible Qdrant versions and remove the unsafe HTTP/API-key warning.
3. Complete one separately approved successful same-run resume and exact ledger settlement.
4. Calibrate automated support signals against representative independent human review.

## P2 — non-blocking follow-ups

- Latency optimization after quality is stable.
- Explicit paid-provider pricing and Decimal cost accounting.
- A separately versioned broader corpus.
- Larger-model comparison only after retrieval quality closes.

Every task defines its hypothesis, implementation, evaluation, acceptance threshold,
dependencies, request/Token planning rule, human-review requirement, risk, rollback, and target
release. Current versioned collections and historical results must remain available for rollback.
