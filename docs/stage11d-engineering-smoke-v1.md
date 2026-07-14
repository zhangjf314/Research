# Stage 11D Engineering-only Bounded Deep Research Smoke

## 1. Scope and non-goals

Stage 11D verifies a bounded Deep Research graph, request and budget accounting, strict citation
validation, isolated run evidence, persistent checkpoints, controlled interruption, and same-run
resume behavior. It is not a Deep Research quality evaluation, production-readiness acceptance, or
v1.0 evidence. No new live request was made during final sealing.

## 2. Fixed smoke manifest

The immutable manifest contains three questions:

- `q003`: single-paper method question.
- `q049`: two-paper comparison question.
- `q005`: unanswerable question that must permit refusal without claims or citations.

The production corpus remains the fixed 34-paper Jina collection. Reranking is disabled.

## 3. Billing and budget model

Billing is explicitly `free`; prices and the monetary budget are exactly zero. The recorded cost
basis is `explicit_free_provider`. Free billing does not bypass request, iteration, token, elapsed,
or reservation limits. Monetary arithmetic uses Decimal.

Per query limits are two iterations, four request attempts, 40,000 tokens, and 300 seconds. Global
limits are twelve request attempts, 120,000 tokens, and 900 seconds. Unknown post-send usage is not
recorded as zero usage: the request retains a conservative active reservation.

## 4. Graph nodes and boundaries

The graph is:

`START -> plan -> retrieve -> assess_evidence -> optional_refine -> synthesize -> validate_citations -> persist_trace -> END`

Only `synthesize` calls the LLM. Provider retries are disabled. Every node boundary, request
lifecycle transition, checkpoint state, and reservation is serializable and persisted.

## 5. Successful live runs

| Question / run | Status | Iterations | Retrieval | Requests | Input | Output | Total | Elapsed s | Claims | Citations | Citation | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q003 / `live-q003-798ac68288e0` | completed | 1 | 1 | 1 | 9,087 | 263 | 9,350 | 8.301317 | 1 | 4 | passed | 0 |
| q049 attempt 3 / `live-q049-4c11db9a2c1d` | completed | 1 | 1 | 1 | 8,959 | 1,010 | 9,969 | 47.425539 | 10 | 10 | passed | 0 |
| q005 / `live-q005-03f669606bb7` | refused | 2 | 2 | 1 | 302 | 35 | 337 | 2.261039 | 0 | 0 | passed | 0 |

The q005 refusal is `answerable=false`, has `claims=[]`, no citations, and the non-empty reason
`No evidence provided to answer the question.`

Formal selected totals are three request attempts, three provider-completed requests, 18,348 input
tokens, 1,308 output tokens, 19,656 total tokens, 57.987895 seconds, and USD 0.

## 6. Provider failure attempts

| Run | Attempt | Parent | Status | Resume | Requests | Settled tokens | Active reservation | Citation | Failure |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `live-q049-24a797337315` | 2 | `live-q049-1d47dc6a1ab8` | provider_failed (migrated historical classification) | 0 | 1 | 0 | 40,000 | not_run | ConnectError |
| `live-q049-03b2d6b68ca3` | 4 | `live-q049-4c11db9a2c1d` | provider_failed | 1 | 1 | 0 | 8,077 | not_run | ConnectError |
| `live-q049-2e5704e44d91` | 5 | `live-q049-03b2d6b68ca3` | provider_failed | 1 | 1 | 0 | 8,077 | not_run | ConnectError |

There are three failed request attempts, three unresolved active reservations, and 56,154
conservatively reserved tokens. Reservations are not provider-reported usage. Attempt 2 uses the
historical per-query ceiling because its original request reservation was not retained.

The additional historical successful q049 attempt 1, `live-q049-1d47dc6a1ab8`, remains available
with 10,026 settled tokens but is not selected by `latest-successful`.

## 7. Run isolation

Every live run has an immutable directory containing `result.json`, `result.csv`, `trace.jsonl`,
`request-ledger.jsonl`, `checkpoint-summary.json`, and `run-metadata.json`. Single-run execution
does not write top-level summaries. All seven directories passed completeness checks, and historical
directory hashes remained unchanged during later attempts.

## 8. Request lifecycle

New requests persist a unique ID and `request_prepared` before provider invocation, then record
`request_started` and either `request_completed` or `request_failed`. Request attempt count,
provider-completed count, and usage-record count are separate. Logical request IDs are globally
unique. No API key or Authorization header is stored.

## 9. Unknown usage accounting

A ConnectError cannot establish whether request bytes reached the provider. Attempts 4 and 5 are
therefore `failed_after_send_unknown` with `unavailable_after_send_attempt`. Each retains its 6,029
input-token estimate plus 2,048 maximum output tokens, for an 8,077-token active reservation. No
usage record or settled token is fabricated, and the failures are not classified as budget blocks.

## 10. Checkpoint and resume findings

Attempts 4 and 5 both passed controlled stop, persistent checkpoint creation, same-run resume,
resume-count increment, completed-node idempotency, retrieval idempotency, request-ID
pre-persistence, lifecycle persistence, failure-trace persistence, conservative reservation, and
duplicate-prevention checks.

Neither resumed provider call returned successfully. Consequently, successful usage settlement,
reservation release, citation validation, and final graph completion after resume remain unverified.
The blocking reason is **BLOCKED BY EXTERNAL PROVIDER CONNECTIVITY**, not checkpoint failure.

## 11. Citation validation

The three selected successful runs passed exact `(paper_id, page, block_id)` triple validation.
No invalid citation is automatically rewritten. q005 emitted no claim or citation. Attempts 4 and 5
did not reach citation validation because their only provider attempts failed.

## 12. Budget compliance

Every individual run stayed under its request, iteration, elapsed, and monetary limits. Across all
seven runs there were seven request attempts, 29,682 settled provider-reported tokens (including
historical q049 attempt 1), 56,154 unresolved conservative reservation tokens, and 222.18158 seconds.
Settled plus reserved tokens equal 85,836, below the 120,000 global ceiling. Total monetary cost is
USD 0. Reservations are reported separately from settled usage.

## 13. Test gates

The reproducible final audit checks run-directory completeness, top-level summary consistency,
SQLite/export consistency, request ledgers, usage sums, failed reservations, duplicate request and
event IDs, q005 citation absence, reranker and Template status, and credential leakage. The final
gates passed: pytest reported 133 passed with one third-party deprecation warning, Ruff reported all
checks passed, compileall completed without output, and `git diff --check` completed without output.

## 14. Limitations

- Successful end-to-end completion after a persisted resume is not verified because two separately
  authorized resumed calls failed with SiliconFlow ConnectError.
- Stage 11D does not measure answer quality, citation precision, or Deep Research report quality.
- Local HTTP Qdrant carries an API key, and qdrant-client 1.18.0 is newer than server 1.12.5. These
  are non-blocking technical debts and are not direct causes of the SiliconFlow failures.

## 15. Final verdict

- Stage 11D three-query live bounded smoke: **PASSED**
- Stage 11D budget guardrails: **PASSED**
- Stage 11D run isolation and attempt accounting: **PASSED**
- Stage 11D provider failure accounting: **PASSED**
- Stage 11D persistent checkpoint and resume idempotency: **PASSED**
- Stage 11D successful end-to-end live resume: **NOT VERIFIED**
- Stage 11D Engineering-only Bounded Smoke: **PARTIALLY PASSED / NOT COMPLETE**
- Stage 11D Deep Research Quality Evaluation: **BLOCKED**
- Production-ready: **NO**
- v1.0: **NOT SATISFIED**
