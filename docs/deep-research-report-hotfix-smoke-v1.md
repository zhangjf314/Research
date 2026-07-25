# Deep Research report hotfix smoke v1

Sanitized metadata only. Raw provider content remains local-only under `.runtime/` and is not included here.

```json
{
  "schema_version": "deep-research-report-hotfix-smoke-v1",
  "legacy_frozen_replay_status": "FAILED",
  "legacy_frozen_replay_classification": "EXPECTED_INCOMPATIBILITY_WITH_REVISED_PROTOCOL",
  "legacy_frozen_replay_gate": "DIAGNOSTIC_ONLY",
  "legacy_failed_smoke": {
    "task_id": "ce25169e-7ab7-4d1b-92f2-fec77df06f0a",
    "live_smoke_status": "FAILED_PROVIDER_SCHEMA",
    "request_attempt_count": 2,
    "provider_completed_request_count": 2,
    "usage_record_count": 2,
    "total_tokens": 13589,
    "estimated_cost_usd": 0.0023809800000000004,
    "raw_response_replay": "FAILED_SECTION_CITATION_CONTRACT",
    "replay_all_passed": false,
    "attempt_failures": [
      {
        "attempt_number": 1,
        "json_parse_status": "passed",
        "normalization_actions": [],
        "schema_error_count": 1,
        "schema_error_locations": [
          "<root>"
        ],
        "schema_error_types": [
          "ValueError"
        ],
        "failure_types": [
          "CITATION_NOT_ALLOWED_FOR_SECTION"
        ],
        "offending_citation_ids": [
          "[E14]",
          "[E2]"
        ],
        "research_synthesis_schema": "failed"
      },
      {
        "attempt_number": 2,
        "json_parse_status": "passed",
        "normalization_actions": [],
        "schema_error_count": 5,
        "schema_error_locations": [
          "research_gaps.0",
          "research_gaps.1",
          "research_gaps.2",
          "research_gaps.3",
          "research_gaps.4"
        ],
        "schema_error_types": [
          "model_type",
          "model_type",
          "model_type",
          "model_type",
          "model_type"
        ],
        "failure_types": [
          "CITATION_NOT_ALLOWED_FOR_SECTION",
          "WRONG_FIELD_TYPE"
        ],
        "offending_citation_ids": [
          "[E14]",
          "[E2]"
        ],
        "research_synthesis_schema": "failed"
      }
    ]
  },
  "live_smoke_status": "PASSED",
  "task_id": "hotfix-deep-research-20260725220952",
  "run_id": "hotfix-deep-research-20260725220952",
  "provider": "deepseek",
  "model": "deepseek-v4-flash",
  "request_attempt_count": 1,
  "provider_completed_request_count": 1,
  "usage_record_count": 1,
  "input_tokens": 7848,
  "output_tokens": 1638,
  "total_tokens": 9486,
  "estimated_cost_usd": 0.00155736,
  "usage_source": "provider_reported",
  "active_reserved_tokens": 0,
  "raw_response_file_count": 1,
  "raw_response_sha256": [
    "172ef3cd1a861a93aaeb37cbea4389f57a860062cd5dbc278a0b501e512b0e39"
  ],
  "raw_response_replay": "PASSED",
  "replay_all_passed": true,
  "schema_validation": "PASSED",
  "citation_validation": "PASSED",
  "research_gap_validation": "PASSED",
  "citation_global_allowlist": "PASSED",
  "citation_section_allowlist": "PASSED",
  "report_quality_gate": "PASSED",
  "exact_duplicate_paragraph_count": 0,
  "normalized_duplicate_bullet_count": 0,
  "duplicate_reference_count": 0,
  "cross_section_similarity": 0.0,
  "raw_provider_content_committed": false,
  "production_deep_research_status": "AVAILABLE",
  "semantic_claim_support_audit": "NOT_FORMALLY_VALIDATED",
  "strong_grounding_claim_allowed": false,
  "retrieval_generalization_evidence": "DIAGNOSTIC_ONLY",
  "commit": "not_created",
  "merge": "not_run",
  "push": "not_run"
}
```
