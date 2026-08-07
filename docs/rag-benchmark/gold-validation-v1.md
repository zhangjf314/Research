# RAG Gold validation v1

- input: `data\evaluation\rag-benchmark\gold-full-v1.jsonl`
- valid: `True`
- records: 146
- approved: 146
- answerable: 132
- unanswerable: 14
- errors: 0
- warnings: 0
- duplicate questions: 0
- near-duplicate questions: 75

Strict structured claim mode is required for the final frozen `rag-gold-v1` dataset. Existing legacy Gold may be audited without this flag while the derived benchmark copy is normalized.
