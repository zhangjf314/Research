# Release Checklist v1.0.0-portfolio

Status: `BLOCKED`

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
- [x] Runtime/API/package version is unified at `0.9.0rc3`.
- [x] Public content claims keep strong generalization disabled.

## Blocking before local release/tag preparation

- [ ] PostgreSQL production checkpoint stop/resume recovery v2 executed.
- [ ] PostgreSQL backup/restore v2 executed against the current runtime.
- [ ] Qdrant snapshot/restore v2 executed with Top-K equivalence comparison.
- [ ] Docker OCR v2 text/mixed/scanned full roundtrip executed.
- [ ] Portfolio 30-minute stability test: `BLOCKED`
- [ ] Git history line-level secret review completed for broad security terms.
- [ ] Docker image rebuilt after adding the OCI version label, then image label
      verified.

## Decision

Do not tag or publish `v1.0.0-portfolio` from this state. The correct current
conclusion is:

`B. Core QA/Deep Research passed, but safety/restore/stability blockers remain.`
