# Stage 4B Runtime Defect Freeze

Status: `BENCHMARK_RUNTIME_DEFECT`

Stage 4B official execution started on branch `eval/research-agent-benchmark-run-v1` at commit `8032e9afdf1a9360467a9dc5b368a6f3017e03ce`.

The run stopped because the global benchmark cost cap was exhausted:

- stop reason: `GLOBAL_BENCHMARK_BUDGET_EXHAUSTED`
- provider requests: `126`
- input tokens: `122010`
- output tokens: `35414`
- total tokens: `157424`
- estimated cost USD: `0.76206156`

The official benchmark is incomplete:

- required official logical runs: `120`
- actual terminal units from unit-level recomputation: `38`
- actual terminal workflow units: `19`
- actual terminal agent units: `19`
- actual complete pairs: `19`
- pending units: `82`

While freezing the incomplete run, a harness summarization defect was found. The generated top-level counters report:

- `official_workflow_runs=19`
- `official_agent_runs=18`
- `complete_pairs=18`
- `workflow_terminal_results=19`
- `agent_terminal_results=18`

Unit-level recomputation from the same result file shows:

- `terminal_workflow=19`
- `terminal_agent=19`
- `complete_pairs=19`

Root cause: the runner checks the global cap at the start of the next execution-unit loop. After the final agent unit was completed and accounted in `global_totals`, the runner encountered the cap before reaching the normal finalization path, so top-level execution counters were not recomputed.

No additional official units were executed after detecting this defect. Stage 4B remains incomplete and Stage 4C is not ready.

This defect does not change the primary conclusion: the benchmark cannot complete under the registered `$0.75` global cost cap.
