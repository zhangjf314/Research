# Stage 4B Failure Validation Protocol Amendment v1

- previous_gate: `DEPLOYED_EXACT_PATH_FAILURE_MATERIALIZATION_REQUIRED`
- revised_gate: `LAYERED_FAILURE_CONTRACT_VALIDATION`
- amendment_hash: `44bf4dc6cc3ce5340f870386d2c87afcd7ad8b66af1589ca073c741626b4ac97`
- layered_gate_passed: `True`
- direct_exact_path_wrong_schema_observed: `False`
- known_validation_limitation: `DEPLOYED_EXACT_PATH_WRONG_SCHEMA_FAILURE_NOT_DIRECTLY_OBSERVED`
- production_failure_injection_hook_added: `False`
- behavior_changed: `False`

## Rationale

Direct deterministic wrong-schema injection through the deployed production HTTP stack would require adding a production test hook after runtime freeze. The amended gate relies on layered evidence without changing Agent, Workflow, RAG, provider configuration, benchmark tasks, rubrics, execution order, evaluation protocol, or budgets.

## Layered gate

- root_cause_established: `True`
- deployed_runtime_source_parity: `True`
- deployed_source_fingerprint_preflight_enabled: `True`
- real_provider_exact_http_path_validation: `True`
- deterministic_failure_materialization_tests: `True`
- controlled_failure_contract_replay: `True`
- provider_usage_on_failure_preserved: `True`
- runner_valid_system_failure_classification: `True`
- agent_behavior_hash_unchanged: `True`
- rag_backend_hash_unchanged: `True`
- workflow_lock_match: `True`
- benchmark_evaluation_hashes_unchanged: `True`
