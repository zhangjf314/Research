# Stage 4 Official Run Invalidation v1

- status: `INVALIDATED_INFRASTRUCTURE`
- invalidated_units: `38`
- invalidated_workflow_units: `19`
- invalidated_agent_units: `19`
- invalidated_complete_pairs: `19`
- provider_requests: `126`
- tokens: `157424`
- cost: `0.76206156`
- quality_results_usable: `False`
- paired_results_usable: `False`
- stage4c_eligible: `False`
- semantic_judge_requests: `0`

## Reason codes

- `WORKFLOW_INVOCATION_CONFIG_DRIFT`
- `BENCHMARK_RUNNER_INVOCATION_FAILURE`
- `BENCHMARK_API_WIRING_FAILURE`
- `UNATTRIBUTABLE_HTTP_503`
- `SUMMARY_COUNTER_DEFECT`

## Rationale

Attempt 1 is not salvageable for paired quality comparison. The runner changed the Workflow invocation contract by passing `max_external_searches=0`, directly causing 12 Workflow failures and potentially affecting other Workflow outcomes. Five HTTP 503 failures also lack response-body/detail capture, so their attribution is not reliable. Agent units are not reused because pairing them with newly executed Workflow units would break the frozen AW/WA temporal paired protocol.
