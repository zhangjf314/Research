# Claim Gold citation comparison v1

Status: `claim_gold_recalculated_diagnostic`

| Experiment | Question exact | Claim macro | Micro core | Core set | Any valid |
|---|---:|---:|---:|---:|---:|
| stage11c_a | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| stage13_2_b | 0.027778 | 0.037037 | 0.040000 | 0.037037 | 0.111111 |
| stage13_3_dev_v2 | 0.127778 | 0.129630 | 0.160000 | 0.111111 | 0.148148 |
| stage13_8_dev_v3_1 | 0.146296 | 0.185185 | 0.200000 | 0.185185 | 0.296296 |

Dev v2 versus Dev v3.1 outcomes: `{'regressed': 1, 'improved': 1, 'unchanged': 7}`.

Focus questions:

```json
{
  "q001": {
    "dev_v2": 0.5,
    "dev_v3_1": 0.0,
    "delta": -0.5,
    "outcome": "regressed"
  },
  "q004": {
    "dev_v2": 0.4,
    "dev_v3_1": 0.4,
    "delta": 0.0,
    "outcome": "unchanged"
  },
  "q015": {
    "dev_v2": 0.0,
    "dev_v3_1": 0.0,
    "delta": 0.0,
    "outcome": "unchanged"
  },
  "q019": {
    "dev_v2": 0.0,
    "dev_v3_1": 0.0,
    "delta": 0.0,
    "outcome": "unchanged"
  },
  "q050": {
    "dev_v2": 0.0,
    "dev_v3_1": 0.0,
    "delta": 0.0,
    "outcome": "unchanged"
  }
}
```

## Focus interpretation

- **q001:** Dev v3.1 increases any-valid evidence through equivalent relations but misses the
  complete two-relation attention/dependency core set. The compound claim remains a decomposition
  candidate.
- **q004:** Both versions hit two of three claim slots, but the GPU/Adam/warmup multi-relation
  training-config set is incomplete. The taxonomy separates not-retrieved from selected-not-cited
  members.
- **q015:** Neither version hits the adjudicated survey-location, ROUGE-limitation, and
  coordinate-ascent-limitation relations. Dev v3.1 contains wrong/insufficient evidence citations.
- **q019:** Neither version hits the exact numeric-range and complete model-shape relations.
  Numeric completeness and retrieval are the dominant failures.
- **q050:** Dev v2 is a fixed-denominator failure. Dev v3.1 hits BERT-side equivalent evidence, so
  any-valid exceeds exact recall, but the cross-paper comparison remains incomplete.

These metrics use AI-assisted manual claim-level Gold for 27 fixed Dev claims. They are diagnostic,
do not replace the Stage 13.8 historical failed Gate or its 0.295 recall, and cannot be extrapolated
to Full-50 or Production.
