# Dev v3.2 Schema Failure Forensics

- Offline only; no raw response, formal result, or historical Gate was changed.
- q005: exact delivered prompt contained the v3.1 examples; classification `model_protocol_version_copy_failure`.
- q007: `finish_reason=length`, 832/832 completion tokens, truncated JSON.
- q013: copied an internal `ev-*` evidence ID from a multi-namespace payload.
- q050: `finish_reason=length`, 832/832 completion tokens, truncated JSON.
- Full prompts and source passages are not reproduced; only hashes and short safe excerpts are retained.

| Question | Failure | Finish | Output tokens | Delivered messages hash |
|---|---|---|---:|---|
| q005 | valid_json_wrong_schema | stop | 50 | `81e8e0bef881ebb78b97c556f1f1659a38b8dc86fd7debfcf150dac020ab2025` |
| q007 | truncated_json | length | 832 | `753cee9a59f4819cff2386c9350c0c2f9d299329f1081e940a199f79fd26970b` |
| q013 | unknown_citation_id | stop | 302 | `4af162467dfb99c8707d60f2ea6cec7a8e41798644a5838447d654a8d9841217` |
| q050 | truncated_json | length | 832 | `5069c0bbf0c066975a5460c04a50a36b134c9cd6da23b55aad2c8e86c3354674` |
