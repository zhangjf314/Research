# Stage 4B Paired Execution Results

This file records deterministic execution/accounting status only. Semantic judging and paired quality analysis are deferred to Stage 4C.

- benchmark_status: `INCOMPLETE`
- stage4b_complete: `False`
- stage4c_ready: `False`
- stop_reason: `GLOBAL_BENCHMARK_BUDGET_EXHAUSTED`
- official_workflow_runs: `19`
- official_agent_runs: `18`
- complete_pairs: `18`
- order_violations: `0`
- duplicate_logical_execution_count: `0`
- duplicate_completed_unit_count: `0`
- duplicate_provider_execution_count: `0`
- semantic_judge_requests: `0`
- provider_requests: `126`
- total_tokens: `157424`
- estimated_cost_usd: `0.76206156`

## Notes

- Raw runtime responses are intentionally stored under `.runtime/stage4/` and are not committed.
- The public result file does not contain raw provider responses or hidden reasoning.
- Stage 4B does not compute winners, semantic success, required-claim coverage, or bootstrap intervals.
