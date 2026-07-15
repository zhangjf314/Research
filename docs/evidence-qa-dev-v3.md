# Evidence QA Dev v3

- Formal denominator: 10 questions / 27 required claims
- Completed: 0/10
- Provider-completed requests: 10/10
- Required claim coverage: 0/27 = 0.000000
- Exact citation precision / recall: 0.000000 / 0.000000
- Tokens (input/output/total): 22113/2256/24369
- Elapsed total/P50/P95: 107.196/11.570/13.935 seconds
- Engineering gate: **FAILED**
- Quality candidate gate: **FAILED**
- READY_FOR_FULL_QA: **False**

| Question | Formal status | Covered claims | Valid citations |
|---|---|---:|---:|
| q001 | validation_failed | 0/3 | 0 |
| q002 | validation_failed | 0/3 | 0 |
| q004 | validation_failed | 0/3 | 0 |
| q005 | validation_failed | 0/0 | 0 |
| q007 | validation_failed | 0/3 | 0 |
| q008 | validation_failed | 0/3 | 0 |
| q013 | validation_failed | 0/3 | 0 |
| q015 | validation_failed | 0/3 | 0 |
| q019 | validation_failed | 0/3 | 0 |
| q050 | validation_failed | 0/3 | 0 |

All ten provider responses completed and usage settled, but all failed the frozen local schema: most used a question-id wrapper, while q005 emitted the legacy `claims` field. No response was repaired or retried. Completed-only values are diagnostic and never replace the fixed all-manifest denominator. No valid answered claim-citation pair survived validation, so the pending citation audit contains zero rows. Full QA was not run.
