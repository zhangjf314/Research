# Claim-first Failure Audit v1

- Claim-first exact availability: **0.125000**
- Routed exact availability: **0.645833**
- Selector: minimum score `0.18`, max `2` per claim
- Context budget: `3000` tokens, section cap `3`
- Unknown roles: `63`, Pilot coverage `18`

The principal deterministic finding is aggressive selector compression: routed Top-10 is reduced to at most two units. A relevance threshold is not evidence-completeness validation. The formal Stage 13 path also uses one query-derived selection claim, so required-claim mappings are not injected and multi-claim budget fragmentation is not the observed cause. All claim-level quality metrics remain unavailable until human Pilot approval.

No Gold block, approved Pilot evidence, or question-specific condition is used by the Production selector.
