# RAG Gold validation v1

- input: `data\evaluation\gold-set-v1.jsonl`
- valid: `False`
- records: 50
- approved: 50
- answerable: 48
- unanswerable: 2
- errors: 10
- warnings: 0
- duplicate questions: 6
- near-duplicate questions: 193

Strict structured claim mode is required for the final frozen `rag-gold-v1` dataset. Existing legacy Gold may be audited without this flag while the derived benchmark copy is normalized.

## Errors

- `duplicate_question`: {'type': 'duplicate_question', 'normalized_question': 'what research problem does the target paper address', 'count': 10}
- `duplicate_question`: {'type': 'duplicate_question', 'normalized_question': 'what are the target paper s main contributions', 'count': 10}
- `duplicate_question`: {'type': 'duplicate_question', 'normalized_question': 'what method or technical approach does the target paper propose', 'count': 10}
- `duplicate_question`: {'type': 'duplicate_question', 'normalized_question': 'how are the target paper s experiments designed and evaluated', 'count': 9}
- `duplicate_question`: {'type': 'duplicate_question', 'normalized_question': 'for the target paper what exact total energy consumption is reported for all experiments', 'count': 2}
- `duplicate_question`: {'type': 'duplicate_question', 'normalized_question': 'what limitations or unresolved issues are reported in the target paper', 'count': 7}
- `unanswerable_has_gold_answer`: {'type': 'unanswerable_has_gold_answer', 'question_id': 'q005'}
- `missing_or_invalid_unanswerable_reason`: {'type': 'missing_or_invalid_unanswerable_reason', 'question_id': 'q005', 'reason': ''}
- `unanswerable_has_gold_answer`: {'type': 'unanswerable_has_gold_answer', 'question_id': 'q030'}
- `missing_or_invalid_unanswerable_reason`: {'type': 'missing_or_invalid_unanswerable_reason', 'question_id': 'q030', 'reason': ''}
