# Stage 4C Metric Provenance

| Metric | Reported value | Provenance | Evidence tier | Safe interpretation |
| --- | --- | --- | --- | --- |
| task_success_rate | see aggregate | STRUCTURAL_PROXY | Tier 2 - Deterministic structural/proxy | Structured task-success proxy, not direct semantic task success. |
| partial_or_better_rate | see aggregate | STRUCTURAL_PROXY | Tier 2 - Deterministic structural/proxy | Partial-or-better structural proxy. |
| required_dimension_coverage | see aggregate | STRUCTURAL_PROXY | Tier 2 - Deterministic structural/proxy | Does not constitute direct scoring of all 250 dimensions. |
| required_claim_coverage | see aggregate | STRUCTURAL_PROXY | Tier 2 - Deterministic structural/proxy | Does not constitute direct semantic scoring of all 180 claims. |
| evidence_coverage | see aggregate | STRUCTURAL_PROXY | Tier 2 - Deterministic structural/proxy | Does not prove each gold evidence set was matched to output text. |
| core_unsupported_claim_rate | see aggregate | STRUCTURAL_PROXY | Tier 2 - Deterministic structural/proxy | A zero value here is not proof of perfect factual reliability. |
| citation_validity | see aggregate | CITATION_STRUCTURE_DERIVED | Tier 2 - Deterministic structural/proxy | Structural validity; empty citation sets can be vacuously valid. |
| gap_handling_accuracy | see aggregate | STRUCTURAL_PROXY | Tier 2 - Deterministic structural/proxy | Structural gap-handling proxy. |
| w_t_l | {'agent': 56, 'tie': 4} | STRUCTURAL_PROXY | Tier 2 - Deterministic structural/proxy | Structured outcome W/T/L, not semantic quality W/T/L. |
| bootstrap_deltas | see aggregate | STRUCTURAL_PROXY | Tier 2 - Deterministic structural/proxy | CI quantifies proxy metric variability, not semantic validity. |

- content_level_rubric_validated: `False`
- structured_proxy_metrics_valid: `True`
