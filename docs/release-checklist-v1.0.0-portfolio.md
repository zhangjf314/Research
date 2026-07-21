# Release Checklist v1.0.0-portfolio

Status: `READY_FOR_LOCAL_RELEASE_COMMIT`

## Passed or preserved

- [x] 50/50 `gold-dev-v1` records are approved.
- [x] Formal QA statistics use approved records only.
- [x] DeepSeek Full QA completed 50/50 with no failures.
- [x] Structured output success is `1.0`.
- [x] Citation ID/context/page validity are `1.0`.
- [x] Deep Research final q003 run completed with strict citation validation.
- [x] Reranker remains disabled.
- [x] Template fallback count is `0`.
- [x] Redis is up and reports real usage.
- [x] Runtime/API/package version is unified at `1.0.0+portfolio` with display
      version `1.0.0-portfolio`.
- [x] Public content claims keep strong generalization disabled.
- [x] Git history line-level secret review completed with `0` confirmed real secrets.
- [x] PostgreSQL checkpoint stop/resume recovery v2 executed.
- [x] PostgreSQL backup/restore v2 executed against the current runtime.
- [x] Qdrant snapshot/restore v2 executed with Top-K equivalence comparison.
- [x] Docker OCR v2 text/mixed/scanned roundtrip executed in the Docker API runtime.
- [x] Portfolio 30-minute stability test executed and passed.

## Final hard gates

- [x] 50 DeepSeek QA: `PASS`
- [x] Deep Research: `PASS`
- [x] Citation ID/context/page: `PASS`
- [x] PostgreSQL checkpoint recovery: `PASS`
- [x] PostgreSQL backup/restore: `PASS`
- [x] Qdrant snapshot/restore: `PASS`
- [x] Redis real usage: `PASS`
- [x] Docker OCR roundtrip: `PASS`
- [x] Git history secret review: `PASS`
- [x] Portfolio 30-minute stability test: `PASS`
- [x] Minimum security audit: `PASS`
- [x] Content claims audit: `PASS`
- [x] Version consistency: `PASS`

## Decision

All local v1.0.0-portfolio hard gates have now passed. The correct current
conclusion is:

`A. All v1.0.0-portfolio hard gates passed; local release preparation is ready, awaiting explicit user authorization for merge/tag/push.`

This checklist does not authorize strong grounding or strong generalization
claims. The semantic status remains:

- `SEMANTIC_CLAIM_SUPPORT_AUDIT=NOT_FORMALLY_VALIDATED`
- `STRONG_GROUNDING_CLAIM_ALLOWED=false`
