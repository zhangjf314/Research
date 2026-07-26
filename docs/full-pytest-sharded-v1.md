# Full Pytest Sharded v1

The full pytest suite passed through deterministic sharded execution. The monolithic pytest command exceeded the execution wrapper timeout and was not used as the final release gate.

- Gate: PASSED
- Collected tests: 677
- Executed tests: 677
- Passed tests: 677
- Failed tests: 0
- Error tests: 0
- Skipped tests: 0
- Shards: 79
- Passed shards: 79
- Failed shards: 0
- Timeout shards: 0
- Missing test files: 0
- Duplicate node IDs: 0
- Total shard runtime seconds: 1182.845

## Slowest shards

- tests/test_stage12_release_readiness.py: 169.163s
- tests/test_stage13_16_dev_v3_4.py: 136.053s
- tests/test_stage13_10_claim_gold.py: 85.335s
- tests/test_stage13_4_claim_coverage.py: 70.256s
- tests/test_stage13_18_payload_contract_v4.py: 60.205s
- tests/test_stage13_17_payload_contract_v3.py: 54.97s
- tests/test_stage13_14_dev_v3_3.py: 49.094s
- tests/test_stage13_19_dev_v3_5_live.py: 48.743s
- tests/test_stage13_3_human_citation_import.py: 47.793s
- tests/test_evidence_qa_dev_v3_1.py: 30.302s
