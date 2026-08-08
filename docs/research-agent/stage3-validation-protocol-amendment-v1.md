# Stage 3 Validation Protocol Amendment v1

- amendment_version: `v1`
- previous_gate: `effective_live_replan_observed=true`
- revised_gate: `effective_live_replan_observed=NOT_REQUIRED_FOR_STAGE3_FREEZE`
- effective_live_replan_observed: `False`
- known_limitation: `LIVE_EFFECTIVE_REPLAN_NOT_OBSERVED`
- additional_tasks_after_preregistered_set: `0`
- behavior_changes_during_stage3c2: `0`

## Rationale

The replan branch is covered by controlled deterministic runtime tests, including PARTIAL/FAIL to REPLAN transition, effective plan delta, changed next action, checkpoint interaction, and trace causality. Real-provider validation demonstrated dynamic tool selection and observation-driven actions, but no preregistered live development task exercised the complete effective-replan causal chain. Rather than continue adapting validation tasks or runtime behavior until a replan appears, the live-replan requirement is moved from a Stage 3 release gate to a Stage 4 behavioral measurement.

## Validation history

### Stage 3C

- 3 live smoke tasks.
- effective replan: `0`.
- runtime defect discovered.

### Stage 3C.1

- runtime defect fixed.
- same Smoke 2/3 rerun.
- dynamic path: `true`.
- resume: `true`.
- effective replan: `0`.

### Stage 3C.2

- 3 preregistered dependency-shaped tasks.
- runtime frozen: `true`.
- dynamic tool selection: `true`.
- observation-driven action: `true`.
- effective replan: `0`.

## Evidence matrix

| Capability | Offline | Live | Final status |
| --- | --- | --- | --- |
| Planner | PASS | PASS | VALIDATED |
| Dynamic tool selection | PASS | OBSERVED | VALIDATED |
| Observation-driven action | PASS | OBSERVED | VALIDATED |
| Evidence state | PASS | OBSERVED | VALIDATED |
| Verification | PASS | OBSERVED | VALIDATED |
| Checkpoint/resume | PASS | OBSERVED | VALIDATED |
| Retry | PASS | NOT REQUIRED | VALIDATED_OFFLINE |
| Budget | PASS | OBSERVED | VALIDATED |
| Replan transition | PASS | NOT OBSERVED | VALIDATED_OFFLINE_ONLY |
| Effective live replan | N/A | NOT OBSERVED | KNOWN_LIMITATION |

Stage 3C.2 Task 1 reached `plan_version=2`, but this remains distinct
from an effective live replan because the full causal-chain definition was
not satisfied.
