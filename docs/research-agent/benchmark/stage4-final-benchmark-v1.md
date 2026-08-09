# Stage 4 Final Workflow vs Agent Benchmark

This report uses only `stage4-official-v1-attempt4` for official quality, reliability, cost, and latency metrics.
Attempts 1 and 2 remain invalidated infrastructure attempts; Attempt 3 remains invalid.

Semantic judging is recorded as a diagnostic gap because the frozen blinded package does not contain fair answer text for both systems.

## Decision matrix

| Dimension | Workflow | Agent | Delta |
| --- | ---: | ---: | ---: |
| Task Success | 0.000 | 0.933 | 0.933 |
| Partial-or-Better | 0.000 | 0.933 | 0.933 |
| Required Claim Coverage | 0.000 | 0.933 | 0.933 |
| Required Dimension Coverage | 0.000 | 0.933 | 0.933 |
| Evidence Coverage | 0.000 | 0.933 | 0.933 |
| Unsupported Claim Rate | 0.000 | 0.000 | 0.000 |
| Failure Rate | 1.000 | 0.067 | -0.933 |
| Tokens/task | 2662.817 | 6909.483 | 4246.667 |
| Cost/task | 0.001 | 0.001 | 0.001 |
| P50 latency | 10.432 | 15.989 | 5.556 |
| P95 latency | 26.836 | 21.447 | -5.389 |

## Win / Tie / Loss

- workflow_wins: `0`
- ties: `4`
- agent_wins: `56`

## Known limitations

- budget_comparable=false; this is a frozen system comparison, not a strict equal-budget causal ablation.
- LIVE_EFFECTIVE_REPLAN_NOT_OBSERVED
- AI semantic judge was not run because fair blind answer text was unavailable in the frozen blind package.
- The benchmark is internally authored/reviewed and should not be described as an independent public benchmark.
