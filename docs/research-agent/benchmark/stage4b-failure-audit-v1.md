# Stage 4B Failure Audit v1

- conclusion: `OFFICIAL_RUN_INVALIDATED_BY_EXECUTION_DEFECT`
- resume_allowed: `False`
- failed_units: `23`
- workflow_failed: `19`
- agent_failed: `4`
- valid_system_failures: `6`
- invalid_infrastructure_failures: `17`
- unknown_failures: `0`
- semantic_judge_requests: `0`

## Failure category counts

- `BENCHMARK_API_WIRING_FAILURE`: `5`
- `BENCHMARK_RUNNER_INVOCATION_FAILURE`: `12`
- `SYSTEM_SCHEMA_FAILURE`: `6`

## Key findings

- `UNIT_RECORDS_ARE_AUTHORITATIVE=true`; unit-level recomputation shows 38 terminal units, 19 workflow terminal units, 19 agent terminal units, and 19 complete pairs.
- The known stale top-level counter defect is summary-only for aggregate reporting.
- However, the official run also contains benchmark invocation failures that affect unit-level results.
- Twelve workflow failures used a runner-supplied `max_external_searches=0`, while the frozen `ResearchBudget` default is `2`; these are not valid Workflow behavior results.
- Five HTTP 503 failures were persisted only as `HTTP Error 503: Service Unavailable`; the runner did not capture response body/detail, so those units cannot be attributed to frozen system behavior.
- Because `invalid_infrastructure_failures > 0`, the current official run must not be resumed for Workflow vs Agent quality conclusions.

## Failed units

| unit | system | status | stop/error | category | validity |
|---|---|---|---|---|---|
| `research-benchmark-v1:rt-v1-001:workflow` | `workflow` | `FAILED` | `research synthesis schema validation failed` | `SYSTEM_SCHEMA_FAILURE` | `valid_system_failure` |
| `research-benchmark-v1:rt-v1-002:agent` | `agent` | `FAILED` | `HTTP_OR_RUNTIME_FAILURE` | `BENCHMARK_API_WIRING_FAILURE` | `invalid_infrastructure_failure` |
| `research-benchmark-v1:rt-v1-002:workflow` | `workflow` | `FAILED` | `max_external_searches` | `BENCHMARK_RUNNER_INVOCATION_FAILURE` | `invalid_infrastructure_failure` |
| `research-benchmark-v1:rt-v1-003:workflow` | `workflow` | `FAILED` | `research synthesis schema validation failed` | `SYSTEM_SCHEMA_FAILURE` | `valid_system_failure` |
| `research-benchmark-v1:rt-v1-004:workflow` | `workflow` | `FAILED` | `research synthesis structured JSON parsing failed` | `SYSTEM_SCHEMA_FAILURE` | `valid_system_failure` |
| `research-benchmark-v1:rt-v1-005:agent` | `agent` | `FAILED` | `HTTP_OR_RUNTIME_FAILURE` | `BENCHMARK_API_WIRING_FAILURE` | `invalid_infrastructure_failure` |
| `research-benchmark-v1:rt-v1-005:workflow` | `workflow` | `FAILED` | `max_external_searches` | `BENCHMARK_RUNNER_INVOCATION_FAILURE` | `invalid_infrastructure_failure` |
| `research-benchmark-v1:rt-v1-006:workflow` | `workflow` | `FAILED` | `max_external_searches` | `BENCHMARK_RUNNER_INVOCATION_FAILURE` | `invalid_infrastructure_failure` |
| `research-benchmark-v1:rt-v1-007:agent` | `agent` | `FAILED` | `HTTP_OR_RUNTIME_FAILURE` | `BENCHMARK_API_WIRING_FAILURE` | `invalid_infrastructure_failure` |
| `research-benchmark-v1:rt-v1-007:workflow` | `workflow` | `FAILED` | `max_external_searches` | `BENCHMARK_RUNNER_INVOCATION_FAILURE` | `invalid_infrastructure_failure` |
| `research-benchmark-v1:rt-v1-008:workflow` | `workflow` | `FAILED` | `max_external_searches` | `BENCHMARK_RUNNER_INVOCATION_FAILURE` | `invalid_infrastructure_failure` |
| `research-benchmark-v1:rt-v1-009:workflow` | `workflow` | `FAILED` | `research synthesis schema validation failed` | `SYSTEM_SCHEMA_FAILURE` | `valid_system_failure` |
| `research-benchmark-v1:rt-v1-010:workflow` | `workflow` | `FAILED` | `max_external_searches` | `BENCHMARK_RUNNER_INVOCATION_FAILURE` | `invalid_infrastructure_failure` |
| `research-benchmark-v1:rt-v1-011:workflow` | `workflow` | `FAILED` | `max_external_searches` | `BENCHMARK_RUNNER_INVOCATION_FAILURE` | `invalid_infrastructure_failure` |
| `research-benchmark-v1:rt-v1-012:agent` | `agent` | `FAILED` | `HTTP_OR_RUNTIME_FAILURE` | `BENCHMARK_API_WIRING_FAILURE` | `invalid_infrastructure_failure` |
| `research-benchmark-v1:rt-v1-012:workflow` | `workflow` | `FAILED` | `max_external_searches` | `BENCHMARK_RUNNER_INVOCATION_FAILURE` | `invalid_infrastructure_failure` |
| `research-benchmark-v1:rt-v1-013:workflow` | `workflow` | `FAILED` | `research synthesis schema validation failed` | `SYSTEM_SCHEMA_FAILURE` | `valid_system_failure` |
| `research-benchmark-v1:rt-v1-014:workflow` | `workflow` | `FAILED` | `max_external_searches` | `BENCHMARK_RUNNER_INVOCATION_FAILURE` | `invalid_infrastructure_failure` |
| `research-benchmark-v1:rt-v1-015:workflow` | `workflow` | `FAILED` | `HTTP_OR_RUNTIME_FAILURE` | `BENCHMARK_API_WIRING_FAILURE` | `invalid_infrastructure_failure` |
| `research-benchmark-v1:rt-v1-016:workflow` | `workflow` | `FAILED` | `max_external_searches` | `BENCHMARK_RUNNER_INVOCATION_FAILURE` | `invalid_infrastructure_failure` |
| `research-benchmark-v1:rt-v1-017:workflow` | `workflow` | `FAILED` | `max_external_searches` | `BENCHMARK_RUNNER_INVOCATION_FAILURE` | `invalid_infrastructure_failure` |
| `research-benchmark-v1:rt-v1-018:workflow` | `workflow` | `FAILED` | `max_external_searches` | `BENCHMARK_RUNNER_INVOCATION_FAILURE` | `invalid_infrastructure_failure` |
| `research-benchmark-v1:rt-v1-019:workflow` | `workflow` | `FAILED` | `research synthesis schema validation failed` | `SYSTEM_SCHEMA_FAILURE` | `valid_system_failure` |
