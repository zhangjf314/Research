# Stage 4B Paired Execution Results

This file records deterministic execution/accounting status only. Semantic judging and paired quality analysis are deferred to Stage 4C.

- official_run_id: `stage4-official-v1-attempt3`
- attempt1_status: `INVALIDATED_INFRASTRUCTURE`
- attempt2_status: `INVALIDATED_INFRASTRUCTURE`
- attempt3_status: `INVALID`
- benchmark_status: `INVALID`
- stage4b_complete: `False`
- stage4c_ready: `False`
- stop_reason: `BENCHMARK_API_WIRING_FAILURE`
- official_workflow_runs: `1`
- official_agent_runs: `2`
- terminal_units: `3`
- pending_units: `117`
- infrastructure_invalid_units: `1`
- complete_pairs: `1`
- order_violations: `0`
- duplicate_logical_execution_count: `0`
- duplicate_completed_unit_count: `0`
- duplicate_provider_execution_count: `0`
- semantic_judge_requests: `0`
- provider_requests: `8`
- total_tokens: `11842`
- estimated_cost_usd: `0.00215782`

## Notes

- Raw runtime responses are intentionally stored under `.runtime/stage4/` and are not committed.
- The public result file does not contain raw provider responses or hidden reasoning.
- Stage 4B does not compute winners, semantic success, required-claim coverage, or bootstrap intervals.
