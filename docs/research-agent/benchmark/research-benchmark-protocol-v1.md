# Research Benchmark Protocol v1

Stage 4 is a paired Workflow vs Agent benchmark. Stage 4A freezes
tasks, rubrics, execution order, fairness constraints, and metrics;
it does not execute either system.

- execution_seed: `40721`
- bootstrap_seed: `41007`
- bootstrap_resamples: `1000`
- concurrency: `1`

## Task success

{
  "mandatory_dimensions_covered": "all",
  "required_claim_coverage_min": 0.8,
  "unsupported_core_claim_count": 0,
  "citation_validity_gate": "passed",
  "gap_task_requires_correct_qualification": true
}

## Replan metrics

- `replan_task_count`
- `effective_replan_count`
- `replan_task_rate`
- `replans_per_task`
- `replan_trigger_distribution`
- `post_replan_success_rate`
- `post_replan_evidence_gain`
- `post_replan_claim_coverage_delta`
