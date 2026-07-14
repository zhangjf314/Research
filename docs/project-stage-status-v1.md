# Project Stage Status v1

Snapshot date: 2026-07-14. Source commit: `e027c3a`. This register separates engineering,
evaluation, and Production status; “passed” in one column never implies the other two.

| Stage | Title | Engineering | Evaluation | Production gate | Main evidence or blocker |
|---|---|---|---|---|---|
| 1 | PDF ingestion and parsing | passed | passed | passed | RC deployment audit |
| 2 | Parser adapters and OCR | partially_passed | passed | partially_passed | Optional Docker OCR engines degraded |
| 3 | Hybrid retrieval and trace | passed | passed | passed | Retrieval baseline artifacts |
| 4 | Paper analysis MVP | passed | passed | passed | Analysis tests and demo cases |
| 5 | External search/import | passed | partially_passed | partially_passed | Semantic Scholar controlled fallback |
| 6 | Research workflow | passed | partially_passed | partially_passed | Quality not established |
| 7 | Evaluation/ablation framework | passed | passed | partially_passed | Early results remain provisional history |
| 8 | Deployment and portfolio | passed | passed | passed | Docker/runbook/documentation |
| 9 | RC real acceptance | passed | partially_passed | passed | `v0.9.0-rc1`; model quality absent |
| 10 | Profiles and human Gold | passed | passed | partially_passed | Gold 50/50 approved; Production gates open |
| 11A | Real Jina Embedding | passed | passed | partially_passed | Engineering passed; initial protocol flawed |
| 11A.5 | Retrieval protocol correction | passed | passed | partially_passed | 34 docs, 2062 points; evidence recall low |
| 11B | Real Jina Reranker | passed | failed | failed | Accepted negative result; default disabled |
| 11C | Real SiliconFlow QA | passed | failed | failed | Schema passed; answer/citation quality failed |
| 11C.5 | QA diagnostics | passed | passed | partially_passed | Token-set signal is not citation correctness |
| 11C.6 | Context optimization | passed | failed | failed | No end-to-end quality improvement |
| 11C.7 | Citation audit | passed | failed | failed | Strict 16.7%, lenient 23.3% on biased sample |
| 11D | Bounded research smoke | passed | not_run | partially_passed | Three engineering smoke outcomes passed |
| 11D.1 | Attempt isolation/resume | partially_passed | not_run | blocked | Resume idempotent; successful resumed completion unverified |

The machine-readable register is
[`data/evaluation/project-stage-status-v1.json`](../data/evaluation/project-stage-status-v1.json).
It contains titles, evidence, metrics, blockers, dependencies, accepted negative results, dates,
and available commits/tags for every row.

The current published maximum is `v0.9.0-rc2`. The current tree satisfies the proposed
`v0.9.0-rc3` strict RC gates and awaits human tag approval; it is not eligible for `v1.0.0`.
