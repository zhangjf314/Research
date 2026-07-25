# Research synthesis current contract v1

This snapshot contains prompt/schema hashes and citation allowlist metadata only. It does not persist full paper evidence text.

```json
{
  "schema_version": "research-synthesis-current-contract-v1",
  "created_at": "2026-07-25T14:01:04.578896+00:00",
  "prompt_version": "deep-research-synthesis-v1",
  "repair_prompt_version": "deep-research-synthesis-v1:repair",
  "research_gap_shape": "object",
  "required_section_ids": [
    "background",
    "methods",
    "results",
    "limitations"
  ],
  "citation_key_format": "E[0-9]{2,3}",
  "section_allowlist_enabled": true,
  "global_allowlist_enabled": true,
  "max_attempts": 2,
  "template_fallback": false,
  "provider": "deepseek",
  "model": "deepseek-v4-flash",
  "response_format": "json_object",
  "delivered_system_prompt_hash": "5c5677d0faec0115f868de09ac8e4070eb39ac89e3c363ccc04484fb8adf2c8b",
  "delivered_user_payload_hash": "ebaeec9a02a5ce8efaad3f1aa7f35f658858de54c6ad8edf09a586820e926dc6",
  "repair_prompt_hash": "3e37e526227bfd00003d1533c104986a0e9423ee4fdcdefe5c9a5905cb49da31",
  "protocol_signature": "7ed08c1a7ce4278a62527c618bdff4905116c793ceb6d3dab974cef413ca0cb5",
  "prompt_snapshot_tests": {
    "user_prompt_contains_required_section_allowlists": true,
    "repair_prompt_contains_required_section_allowlists": true,
    "user_prompt_contains_research_gap_object_skeleton": true,
    "repair_prompt_contains_research_gap_object_skeleton": true
  },
  "section_allowlist_preflight": {
    "path": "data/evaluation/research-section-allowlist-preflight-v1.json",
    "mapping_invariant_errors": [],
    "unassigned_evidence_ids": [],
    "global_evidence_count": 16
  }
}
```

## Section allowlist preflight

```json
{
  "schema_version": "research-section-allowlist-preflight-v1",
  "created_at": "2026-07-25T14:01:12.718533+00:00",
  "query": "RAG 方法的主要技术路线、实验结果和局限分别是什么？",
  "global_evidence_count": 16,
  "background_allowed_ids": [
    "E01",
    "E02",
    "E03",
    "E04",
    "E05",
    "E06",
    "E07",
    "E08"
  ],
  "methods_allowed_ids": [
    "E09",
    "E01",
    "E10",
    "E11",
    "E04",
    "E12",
    "E06",
    "E05"
  ],
  "results_allowed_ids": [
    "E01",
    "E03",
    "E04",
    "E13",
    "E06",
    "E07",
    "E08",
    "E12"
  ],
  "limitations_allowed_ids": [
    "E01",
    "E14",
    "E02",
    "E15",
    "E04",
    "E06",
    "E16",
    "E08"
  ],
  "multi_section_evidence_ids": [
    "E01",
    "E02",
    "E03",
    "E04",
    "E05",
    "E06",
    "E07",
    "E08",
    "E12"
  ],
  "unassigned_evidence_ids": [],
  "mapping_invariant_errors": [],
  "model_visible_key_format": "E01",
  "full_evidence_text_persisted": false,
  "llm_called": false,
  "reranker_enabled": false
}
```
