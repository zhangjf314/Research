# Soak Test Portfolio v1

Status: `BLOCKED`

Stage 13.39 requires a Portfolio 30-minute stability test as the portfolio
release hard gate.

## Configuration

```dotenv
SOAK_DURATION_SECONDS=1800
SOAK_MAX_LLM_REQUESTS=8
SOAK_MAX_TOTAL_TOKENS=80000
SOAK_MAX_COST_USD=0.05
SOAK_LLM_SAMPLE_INTERVAL_SECONDS=300
```

## Required workload

The test must include:

1. Periodic retrieval.
2. Redis cache hit/miss activity.
3. PostgreSQL queries.
4. Qdrant queries.
5. 3-5 short real QA calls.
6. 1 short Deep Research run.
7. 1 Docker OCR roundtrip.
8. 1 API container recreation.
9. Post-restart health, capabilities, checkpoint, and reservation checks.

## Pass gate

- `duration_seconds >= 1800`
- `fatal_error_count = 0`
- `unclassified_exception_count = 0`
- `api_restart_count = 1`
- `api_restart_recovery = passed`
- `postgres_available = true`
- `qdrant_available = true`
- `redis_available_and_used = true`
- `active_reserved_tokens = 0`
- `checkpoint_consistency = passed`
- `qa_success_rate >= 0.95`
- `deep_research_success_count >= 1`
- `ocr_roundtrip = passed`

## Required measurements

- request count, success count, failure count
- P50/P95/P99 latency
- API RSS and API CPU
- container memory
- PostgreSQL connection count
- Redis key count and cache hit rate
- Qdrant point count
- checkpoint count
- input/output/total tokens
- estimated cost
- restart recovery time

This 1800-second stability test was not executed here. Existing earlier stability
evidence is useful historical evidence, but it is not a substitute for the
Portfolio 30-minute stability test gate.

Allowed memory interpretation after a passing run:

> Within this 30-minute test window, no obvious sustained abnormal memory growth
> was observed.

Forbidden interpretations:

- proof that memory leaks do not exist
- claims that the test proves stability beyond the measured 30-minute window
- claims that the test is equivalent to a commercial endurance program

No Full QA rerun and no successful Deep Research rerun was performed in this
Stage 13.39 audit.
