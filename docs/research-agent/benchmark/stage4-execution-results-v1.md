# Stage 4B Paired Execution Results

This file records deterministic execution/accounting status only. Semantic judging and paired quality analysis are deferred to Stage 4C.

- official_run_id: `stage4-official-v1-attempt4`
- attempt1_status: `INVALIDATED_INFRASTRUCTURE`
- attempt2_status: `INVALIDATED_INFRASTRUCTURE`
- attempt3_status: `INVALID`
- attempt4_status: `VALID_COMPLETE`
- benchmark_status: `COMPLETE`
- stage4b_complete: `True`
- stage4c_ready: `True`
- stop_reason: `ALL_UNITS_TERMINAL`
- official_workflow_runs: `60`
- official_agent_runs: `60`
- workflow_terminal_results: `60`
- agent_terminal_results: `60`
- terminal_units: `120`
- pending_units: `0`
- infrastructure_invalid_units: `0`
- complete_pairs: `60`
- order_violations: `0`
- duplicate_logical_execution_count: `0`
- duplicate_completed_unit_count: `0`
- duplicate_provider_execution_count: `0`
- semantic_judge_requests: `0`
- provider_requests: `487`
- provider_failures: `0`
- input_tokens: `467210`
- output_tokens: `107128`
- total_tokens: `574338`
- estimated_cost_usd: `0.09540524`

## Attempt history

| Attempt | Status | Run ID | Terminal units | Pending units | Infrastructure invalid units | Paired results usable | Stage 4C eligible | Cost USD |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| 1 | `INVALIDATED_INFRASTRUCTURE` | `a34f8217b923c93d` | `38` | `n/a` | `n/a` | `False` | `False` | `0.76206156` |
| 2 | `INVALIDATED_INFRASTRUCTURE` | `stage4-official-v1-attempt2` | `3` | `117` | `1` | `False` | `False` | `0.05137732` |
| 3 | `INVALID` | `stage4-official-v1-attempt3` | `3` | `117` | `1` | `False` | `False` | `0.00215782` |
| 4 | `VALID_COMPLETE` | `stage4-official-v1-attempt4` | `120` | `0` | `0` | `True` | `True` | `0.09540524` |

## Notes

- Attempts 1 and 2 remain invalidated for infrastructure reasons and are excluded from Stage 4C.
- Attempt 3 remains invalid due to benchmark API wiring failure and is excluded from Stage 4C.
- Attempt 4 is the only valid complete Stage 4B paired execution and is the only source for the blinded Stage 4C package.
- Raw runtime responses are intentionally stored under `.runtime/stage4/` and are not committed.
- The public result file does not contain raw provider responses or hidden reasoning.
- Stage 4B does not compute winners, semantic success, required-claim coverage, or bootstrap intervals.
