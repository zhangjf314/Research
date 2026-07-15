# Dev v2 Required Claim Coverage Audit

- Required claims: 27
- Covered: 14
- Omitted: 10
- Merged candidates: 3
- Unsupported before generation: 6

## Failure stages

- covered: 14
- malformed_json: 3
- model_omitted_claim: 3
- required_claim_matching_failure: 3
- retrieval_ranked_out: 4

## Per question

| Question | Required | Covered | Omitted | Merged |
|---|---:|---:|---:|---:|
| q001 | 3 | 1 | 1 | 1 |
| q002 | 3 | 2 | 0 | 1 |
| q004 | 3 | 2 | 0 | 1 |
| q005 | 0 | 0 | 0 | 0 |
| q007 | 3 | 3 | 0 | 0 |
| q008 | 3 | 3 | 0 | 0 |
| q013 | 3 | 1 | 2 | 0 |
| q015 | 3 | 0 | 3 | 0 |
| q019 | 3 | 2 | 1 | 0 |
| q050 | 3 | 0 | 3 | 0 |

Required claims were not explicitly represented in the v2 prompt payload. Candidate triples were not fully materialized in the historical trace, so candidate-missing versus ranked-out cannot always be separated; this limitation is preserved rather than inferred.
