# Provider Schema Reliability Audit v1

This audit covers the unreleased `v1.1.1` reliability patch series on
`fix/v1.1.1-provider-schema-reliability`. It does not change Stage 4 benchmark
results, which remain tied to `v1.1.0-portfolio`.

## Failure history

| Round | Failure | Root cause type | Patch status |
| --- | --- | --- | --- |
| R1 | Section citation ownership mismatch | Cross-field model constraint | Diagnostics added |
| R2 | Cross-section citation misuse | Section-scoped citation contract | Targeted section repair added |
| R3 | Duplicate report bullet | Content-quality relation | Duplicate forensics and targeted repair added |
| R4 | `insufficient_evidence=true` with non-empty claims | Cross-field state contradiction | Derived-state compiler added |

## Latest failure

- task_id: `015f3f64-233a-4b88-baf7-6a4686005427`
- section: `results`
- claims_count: `2`
- evidence_gap_count: `1`
- provider-authored `insufficient_evidence`: `true`
- citations: `E01`, `E03`
- citations known: `true`
- citations in section allowlist: `true`
- schema/citation/duplicate validation after deterministic derivation: `passed`
- new provider requests for this audit: `0`

The latest failure was not a citation ownership failure and not a duplicate
quality failure. The provider returned useful supported claims and a legitimate
gap, but also emitted a contradictory boolean state.

## `insufficient_evidence` semantics

The domain meaning is narrow:

> A final section has no supported claims and must expose an evidence gap.

The provider-facing meaning is weaker:

> A non-authoritative hint that may indicate missing or partial evidence.

This makes the boolean too coarse for partial evidence. A section can contain
supported claims while still having an evidence gap for missing experimental
coverage. R4 preserves that gap by promoting it to top-level `research_gaps`
when compiling the provider draft.

## Current truth table

| Claims | Evidence gap | Provider `insufficient_evidence` | Pre-R4 domain legal | R4 derived state |
| --- | --- | --- | --- | --- |
| present | absent | false | yes | supported |
| present | present | false | no | supported; gap preserved as research gap |
| present | present | true | no | supported; gap preserved as research gap |
| absent | present | true | yes | insufficient |
| absent | present | false | no | insufficient |
| absent | absent | true | no | invalid |

## Provider schema field provenance

### Content generated

- `title`
- `executive_summary`
- `section.summary`
- `section.claims[].text`
- `section.claims[].citation_ids`
- `consensus[]`
- `disagreements[]`
- `research_gaps[]`

### Model judgment

- `section.evidence_gap`
- `research_gaps[]`

### Derivable state

- `section.insufficient_evidence`

### Redundant state

- `section.insufficient_evidence` vs `section.claims`
- `section.insufficient_evidence` vs `section.evidence_gap`

### System control state

- schema parse status
- validation errors
- report quality
- terminal status
- usage and cost accounting

## Multiple sources of truth

The provider-facing contract previously required the model to maintain three
related facts:

1. whether section claims exist;
2. whether a section evidence gap exists;
3. whether `insufficient_evidence` is true.

This is a multiple-sources-of-truth reliability risk. R4 makes the final domain
state deterministic from the validated content structure.

## Architecture decision

Selected option: **Option A — patch-safe internal derived-state fix**.

Reasoning:

- public API remains unchanged;
- semantic quality gates remain unchanged;
- citation validation remains strict;
- duplicate quality gate remains strict;
- provider still authors claims, citations, and gaps;
- system derives only deterministic section state;
- latest failure can be safely derived because claims and citations validate.

The implementation introduces an internal provider draft compilation step:

```text
ProviderSynthesisDraft
  -> deterministic compile / derive state
  -> ResearchSynthesis
  -> citation validation
  -> duplicate/report validation
```

The compiler is not allowed to create claims, invent evidence, replace
citations, delete unsupported claims, or hide validation failures.

## Public API impact

- `PUBLIC_API_SCHEMA_CHANGE_REQUIRED=false`
- `semantic_behavior_change_required=false`
- `provider_schema_equals_domain_schema_before_r4=true`
- `compiler_layer_recommended=true`

## Release status

This audit and implementation do not make `v1.1.1` release-ready by themselves.
The release gate still requires a fresh deployed 5/5 exact UI sample smoke after
offline gates and source parity pass.

- `benchmark_rerun=false`
- `stage4_results_unchanged=true`
- `v1.1.1_release_ready=false`
