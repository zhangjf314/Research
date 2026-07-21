# Dev v3.1 Citation Recall Audit

- Required-claim rows: 27
- Formal Dev v3.1 recall: 0.295000
- Answerable-only macro question recall (diagnostic): 0.216667
- Macro required-claim recall (diagnostic): 0.135185
- Micro question-gold-block recall (diagnostic): 5/33 = 0.151515
- Dev v2 published-path reconstruction: 0.295833
- Per-question formal recalls: `{"q001": 0.0, "q002": 1.0, "q004": 0.2, "q005": 1.0, "q007": 0.25, "q008": 0.25, "q013": 0.25, "q015": 0.0, "q019": 0.0, "q050": 0.0}`
- Duplicate citations within claim / shared citations across claims: 0/7
- Exact/page hit instances across claim rows: 9/20
- Dev v3.1 formula: mean of all ten fixed per-question recalls; q005 refusal contributes 1 and q050 contributes 0.
- Dev v2 formula: mean over eight completed answerable questions; q005 is excluded as unanswerable and failed q050 is excluded.
- Decision: **CITATION_RECALL_METRIC_INCOMPARABLE**. The 0.295000 versus 0.295833 difference is caused by denominator/protocol differences, not float serialization or rounding.
- Formal Stage 13.8 metric and gate were not changed.
- Citation audit schema/hash/triple validation: {'records': 33, 'unique_sample_ids': 33, 'pending': 33, 'source_hash_valid': True, 'source_record_hash_valid': True, 'immutable_hash_valid': True, 'registry_hash_valid': True, 'citation_triples_valid': True}
