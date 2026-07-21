# q017 Live Retry Config v1

- Git commit: `98058ceef882b317afa9b6f2086b9da9ffdac3d0`
- Provider/model: `siliconflow` / `Qwen/Qwen3-8B`
- Prompt: `qa-production-v1`
- Response format: `json_object`
- Provider retry count: `0`
- JSON repair enabled: `false`
- QA retry count: `0`
- Response audit enabled: `true`
- Live retry gate: `{'QA_RESPONSE_AUDIT_ENABLED': True, 'sanitized_response_audit_persistence': 'verified_by_unit_test_and_container_config', 'q017_context_analysis_completed': True, 'retrieval_fix_regression_tests_passed': True, 'q017_gold_or_valid_equivalent_in_context': True, 'live_model_budget_ready': True}`
