# Citation Recall Metric v2

- Protocol: `citation-recall-v2`
- Primary: `answerable_question_macro_exact_recall_v2`
- Fixed denominator: 9 answerable questions
- q005: excluded from citation recall
- q050/provider/schema/validation failure: recall 0, never dynamically excluded
- Duplicate handling: exact full relation key
- Historical Stage 13.8 Gate effect: none; it remains FAILED.
- Frozen before any Dev v3.2 live run: true
