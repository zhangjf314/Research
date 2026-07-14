# Known Issues v1

The complete structured register is
[`data/evaluation/known-issues-v1.json`](../data/evaluation/known-issues-v1.json).

| ID | Severity | Issue | Release effect |
|---|---|---|---|
| KI-001 | critical | Exact evidence availability is insufficient | v1 blocker |
| KI-002 | critical | Citation support: 16.7% strict, 23.3% lenient | v1 blocker |
| KI-003 | high | Context optimization did not improve end-to-end quality | v1 blocker |
| KI-004 | high | Jina reranker failed enablement gate | Keep disabled; not a blocker while disabled |
| KI-005 | critical | Deep Research quality was not evaluated | v1 blocker |
| KI-006 | high | Successful resumed live completion unverified | v1 blocker |
| KI-007 | high | Repeated SiliconFlow ConnectError | External v1 blocker |
| KI-008 | high | Qdrant HTTP connection carries API-key warning | v1 security blocker |
| KI-009 | high | qdrant-client/server compatibility warning | v1 operations blocker |
| KI-010 | medium | Citation audit is AI-assisted and failure-stratified | Calibration limitation |
| KI-011 | low | First-token latency unavailable in non-streaming JSON mode | Disclosed limitation |
| KI-012 | medium | Explicit-free SiliconFlow policy may change | Cost-model limitation |

The Qdrant warnings are not the direct cause of SiliconFlow connectivity failures. Active token
reservations on failed requests are conservative unknown-usage accounting, not Provider-reported
usage.
