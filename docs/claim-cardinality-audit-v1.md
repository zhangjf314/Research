# Claim Cardinality Audit v1

This audit checks whether the Direct QA maximum of 3 generated claims creates a mathematical coverage ceiling.

## gold_dev_v1

- records: `50`
- distribution: `{3: 48, 1: 2}`
- mean/median/p95/max: `2.92` / `3.0` / `3` / `3`
- count(required_claim_count > 3): `0`
- theoretical coverage cap with max 3 generated claims and one-to-one matching: `1.0`

## canary_15

- records: `15`
- distribution: `{3: 14, 1: 1}`
- mean/median/p95/max: `2.866667` / `3` / `3` / `3`
- count(required_claim_count > 3): `0`
- theoretical coverage cap with max 3 generated claims and one-to-one matching: `1.0`

## Conclusion

The existing Direct QA prompt caps generated claims at 3. If the evaluator only allowed strict one-to-one matching, questions with more than 3 required claims would have a mathematical coverage ceiling. Current gold-dev-v1 records all have three or fewer required claims, so the low canary coverage is not caused by this cardinality cap in the current dataset.
