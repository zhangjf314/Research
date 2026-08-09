# Portfolio Release Status v1

## Status

- Release target: `v1.1.0-portfolio`
- Package/runtime: `1.1.0+portfolio`
- Release status: ready for final merge, tag, and push after local gates pass
- GitHub Release creation: not authorized in this task

## Passed evidence retained

- Full QA: 50/50 completed with real DeepSeek calls.
- Production Deep Research: available after structured synthesis/reporting
  hotfixes and live smoke/replay validation.
- Operations: PostgreSQL checkpoint recovery, PostgreSQL backup/restore, Qdrant
  snapshot/restore, Docker OCR roundtrip, Redis production recheck, and the
  Portfolio 30-minute stability test passed.
- Stage 4 Workflow vs Agent: final Attempt 4 benchmark and Stage 4C validity
  audit completed without rerunning live systems in the validity step.

## Stage 4 release interpretation

`READY_WITH_SEMANTIC_EVALUATION_LIMITATION`

The structured proxy metrics are valid for portfolio-level engineering claims.
The release does not include a formal semantic judge or human content-level
rubric pass over all compared outputs.

## Tag policy

- Keep `v1.0.0-portfolio` unchanged.
- Keep `v1.0.1-portfolio` unchanged.
- Create `v1.1.0-portfolio` as a new annotated tag only after the final release
  tests pass on `main`.

## Forbidden claims

- `STRONG_GENERALIZATION_CLAIM_ALLOWED=false`
- `STRONG_GROUNDING_CLAIM_ALLOWED=false`
- `SEMANTIC_BENCHMARK_CLAIM_ALLOWED=false`
- `COMMERCIAL_PRODUCTION_READY_CLAIM_ALLOWED=false`
