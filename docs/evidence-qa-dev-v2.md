# Evidence QA Dev v2

- Manifest: `fcb59b71fc68549479c24f6475f7d18ad9e382aace93e70e93594ee355ffb988`
- Runs: 10/10; completed: 9; post-processing failures: 1
- Requests / Provider completions / usage records: 10 / 10 / 10
- Settled input / output / total tokens: 37594 / 5393 / 42987
- Active reservations: 0; retained historical reservations: 60000
- Total elapsed: 237.094871 s; all-attempt P95: 83711.721 ms
- Monetary cost: 0 USD (`explicit_free_provider`)

## Formal all-manifest metrics

- Schema success: 0.9
- Answerable / refusal accuracy: 0.888889 / 1.0
- Required claim coverage: 0.518519
- Exact citation precision / citation recall: 0.181731 / 0.295833
- Page citation precision / claim-citation binding: 0.437179 / 1.0
- Unsupported claim rate: 0.8
- Unknown citation ID / invalid citation rate: 0 / 0.0
- Context tokens mean / retrieval P95: 2063.0 / 30.4089 ms

## Per-question comparison to historical Stage 11C

| Question | Status | Classification | Coverage delta | Precision delta | Recall delta |
|---|---|---|---:|---:|---:|
| q001 | completed | improved | 0.000000 | 0.400000 | 0.666667 |
| q002 | completed | improved | 0.333333 | 0.153846 | 1.000000 |
| q004 | completed | regressed | -0.333333 | -0.085714 | -0.200000 |
| q005 | completed | unchanged_or_mixed | 0.000000 | 0.000000 | 0.000000 |
| q007 | completed | improved | 0.000000 | 0.200000 | 0.250000 |
| q008 | completed | improved | 0.666667 | 0.500000 | 0.250000 |
| q013 | completed | regressed | -0.333333 | 0.000000 | 0.000000 |
| q015 | completed | regressed | -0.333333 | -0.250000 | -0.333333 |
| q019 | completed | improved | 0.666667 | 0.000000 | 0.000000 |
| q050 | validation_failed | regressed | -1.000000 | 0.000000 | 0.000000 |

Changes: improved=5, regressed=4, unchanged/mixed=1; Phase-B gain queries improved=2/4.
Different historical schemes have different execution denominators; these deltas are diagnostic and do not establish statistical superiority.

## Gates

- Engineering gate: **True**
- Quality candidate gate: **False**
- Failed quality condition: required claim coverage did not remain at or above 0.555556.
- READY_FOR_FULL_QA: **False**
- Full QA run: **False**
- Deep Research run: **False**
- Pending citation audit samples: 57

Formal metrics use the fixed 10-question denominator; completed-only metrics are diagnostic.
