# Dev v3 Response Shape Audit

- Records: 10
- Valid JSON: 10/10
- Schema families: `{'question_id_wrapper': 7, 'required_claim_id_map': 1, 'legacy_refusal': 1, 'legacy_v2_claims': 1}`
- Deterministic normalization possible: 0/10

| Question | Family | Top-level keys | Normalizable |
|---|---|---|---|
| q001 | question_id_wrapper | ['q001'] | False |
| q002 | required_claim_id_map | ['cl-q002-023231b350a389faa754', 'cl-q002-3cab1a55474dcd47a64a', 'cl-q002-8a1d729edcaafb379a20'] | False |
| q004 | question_id_wrapper | ['q004'] | False |
| q005 | legacy_refusal | ['question_id', 'answerable', 'claims', 'refusal_reason'] | False |
| q007 | question_id_wrapper | ['q007'] | False |
| q008 | question_id_wrapper | ['q008'] | False |
| q013 | legacy_v2_claims | ['question_id', 'claims', 'answerable'] | False |
| q015 | question_id_wrapper | ['q015'] | False |
| q019 | question_id_wrapper | ['q019'] | False |
| q050 | question_id_wrapper | ['q050'] | False |

Official status remains `validation_failed` for every record.
