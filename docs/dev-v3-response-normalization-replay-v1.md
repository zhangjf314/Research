# Dev v3 Response Normalization Replay

- Strict raw schema success: 0/10
- Deterministically normalized schema success: 0/10
- Strict/diagnostic coverage: 0/27 / 0/27
- Rejections: 10

| Question | Operations considered | Status | Schema pass |
|---|---|---|---|
| q001 | ['single_exact_question_wrapper_unwrap'] | normalization_rejected | False |
| q002 | [] | normalization_rejected | False |
| q004 | ['single_exact_question_wrapper_unwrap'] | normalization_rejected | False |
| q005 | [] | legacy_semantic_salvage_only | False |
| q007 | ['single_exact_question_wrapper_unwrap'] | normalization_rejected | False |
| q008 | ['single_exact_question_wrapper_unwrap'] | normalization_rejected | False |
| q013 | [] | legacy_semantic_salvage_only | False |
| q015 | ['single_exact_question_wrapper_unwrap'] | normalization_rejected | False |
| q019 | ['single_exact_question_wrapper_unwrap'] | normalization_rejected | False |
| q050 | ['single_exact_question_wrapper_unwrap'] | normalization_rejected | False |

This is diagnostic replay only. It neither replaces Stage 13.5 nor permits Full QA. No semantic repair, fuzzy matching, missing-slot creation, free-triple conversion, or LLM call occurred.
