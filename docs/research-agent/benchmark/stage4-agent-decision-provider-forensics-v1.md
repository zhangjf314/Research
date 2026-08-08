# Stage 4B.3 AgentDecisionProviderError Forensics

## Status

- Attempt1: `INVALIDATED_INFRASTRUCTURE`
- Attempt2 before this audit: `INVALID`
- Attempt2 after this audit: `INVALIDATED_INFRASTRUCTURE`
- Stage4B complete: `false`
- Stage4C ready: `false`
- Attempt3 started: `false`

Attempt2 is not resumable and its 3 terminal units are not salvageable for official quality results.

## Failed unit

- unit: `research-benchmark-v1:rt-v1-002:agent`
- runtime task id before isolation fix: `stage4-rt-v1-002-agent`
- HTTP status seen by runner: `503`
- runner detail code: `HTTP_503`
- runner detail message: `research agent failed: AgentDecisionProviderError`
- request_id: `57778858-ae75-4bc3-baa2-4df2d1a26599`

The runner did not receive a normal agent terminal response, trace summary, verification state, or provider usage for this unit.

## Offline forensics result

Container-local artifacts were found for the failed task:

- provider audit: found
- checkpoint: found
- trace: found
- raw provider response: privately persisted, not committed

The real provider request did occur and completed:

- provider: `deepseek`
- model: `deepseek-v4-flash`
- provider HTTP status: `200`
- provider_request_id: `null`
- input tokens: `491`
- output tokens: `599`
- total tokens: `1090`
- usage_source: `provider_reported`
- estimated_cost_usd: `0.00023646000000000002`
- retry_count: `0`

The provider returned valid JSON, but the JSON echoed the request payload instead of returning the required `ResearchPlan` shape. The decision provider therefore raised:

- root_exception: `AgentDecisionProviderError`
- root_error_code: `DECISION_SCHEMA_VALIDATION_ERROR`
- subtype: `PROVIDER_VALID_JSON_WRONG_SCHEMA`
- effective message: `invalid provider plan payload: KeyError`

## Expected vs actual failure contract

Frozen Agent already defines `PROVIDER_FAILURE` as a stop reason. A permanent decision-provider failure should be materialized as:

- status: `FAILED`
- stop_reason: `PROVIDER_FAILURE`
- failure_code: `AGENT_DECISION_PROVIDER_ERROR`
- verification_status: `N/A` when the verifier never ran
- provider usage: retained if the provider returned usage

Actual pre-fix path:

1. Provider returned HTTP 200 and usage.
2. Decision provider rejected the JSON shape.
3. `AgentDecisionProviderError` was raised.
4. Runner persisted provider usage.
5. Runner generic exception path marked `stop_reason=CHECKPOINT_FAILURE`.
6. API catch-all converted the error to HTTP 503.
7. Benchmark runner classified the unit as `BENCHMARK_API_WIRING_FAILURE`.

Classification:

```text
AGENT_FAILURE_MATERIALIZATION_DEFECT
```

This is an observability/API-contract defect, not an Agent planner/policy/verifier behavior defect.

## Fix scope

Behavior-neutral fixes applied:

- materialize `AgentDecisionProviderError` as terminal `FAILED + PROVIDER_FAILURE`;
- expose `failure_code=AGENT_DECISION_PROVIDER_ERROR` in the Agent API response;
- flush an attributable checkpoint and trace event for provider failure;
- classify `PROVIDER_FAILURE` as `SYSTEM_PROVIDER_FAILURE / valid_system_failure` in the Stage4 runner;
- namespace benchmark API task ids by official run id to prevent cross-attempt checkpoint/trace collisions.

Not changed:

- Agent planner
- Agent policy
- Agent verifier
- decision prompt
- decision schema
- parser tolerance
- provider retry count
- tool registry
- RAG backend
- Workflow behavior
- benchmark tasks
- rubrics
- evaluation protocol

## Hash and lock status

- stage3_agent_behavior_hash before: `bce71a51171b2e1187d579a2278cc34f1202ed7b84e9482cbffe42d00b92ff15`
- stage3_agent_behavior_hash after: `bce71a51171b2e1187d579a2278cc34f1202ed7b84e9482cbffe42d00b92ff15`
- agent behavior hash match: `true`
- rag backend hash: `995a144385180b2931ec2c6366f7f7306301a42d77ad7c85f4be9e6d9e5091d9`

The fix does not edit `src/paper_research/agents/research_agent/*`, so the frozen Stage 3 source-backed agent behavior lock remains unchanged.

## Attempt2 disposition

Attempt2 remains invalidated:

- do not resume Attempt2;
- do not salvage completed units;
- do not use Attempt2 for Stage4C;
- if later authorized, Attempt3 must start from 0/120 in a new runtime directory.

## Next gate before Attempt3

Before any Attempt3 authorization, run non-benchmark live wiring validation using excluded Stage 3 development tasks only. Required result:

- infrastructure_failures: `0`
- usage_accounting_gaps: `0`
- summary_unit_mismatch: `0`
- duplicate_execution: `0`
- agent_behavior_hash_match: `true`
- workflow_lock_match: `true`
- rag_lock_match: `true`

This validation was not run in Stage 4B.3 because `stage4-task-exclusions-v1` was not present in the repository. Official benchmark tasks were not used as substitutes.

No Attempt3 was started in this phase.
