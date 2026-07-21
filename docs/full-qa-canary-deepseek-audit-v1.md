# full-qa-canary-deepseek-v1 Audit

- Provider/model: `deepseek` / `deepseek-v4-flash`
- Gate: `FAILED`
- Attempted/completed/failed: `15` / `15` / `0`
- Malformed/schema/invalid citation: `0` / `0` / `0`
- Citation context validity: `1.0`
- Page accuracy: `1.0`
- Core unsupported claim count: `30`
- Finish reason length count: `0`
- Required claim coverage: `0.357143`
- Citation precision / recall: `0.202381` / `0.139286`
- Tokens input/output/total: `154876` / `2259` / `157135`
- Estimated cost USD: `0.02231516`
- Cost type: `estimated_upper_bound_from_configured_prices`
- Budget violations: `[]`

This canary is an internal development gate, not a blind benchmark.
The cost was recalculated deterministically from provider token usage and configured per-million token prices; no model request was rerun.
