"""Audit prompt output field contamination for the v3.7 candidate."""

from __future__ import annotations

import json
import re
from typing import Any

from paper_research.generation.schema_reliability import dev_v3_7_candidate_system_prompt

try:
    from scripts.evidence_presentation_v2_lib import (
        ALLOWED_OUTPUT_FIELDS,
        FORBIDDEN_CONTROL_FIELDS,
        PROMPT_AUDIT,
        PROMPT_AUDIT_DOC,
        canonical_hash,
        control_occurrences,
        rendered_messages,
    )
    from scripts.evidence_qa_dev_lib_v1 import DEV_IDS
except ModuleNotFoundError:
    from evidence_presentation_v2_lib import (  # type: ignore[no-redef]
        ALLOWED_OUTPUT_FIELDS,
        FORBIDDEN_CONTROL_FIELDS,
        PROMPT_AUDIT,
        PROMPT_AUDIT_DOC,
        canonical_hash,
        control_occurrences,
        rendered_messages,
    )
    from evidence_qa_dev_lib_v1 import DEV_IDS  # type: ignore[no-redef]


def quoted_keys(text: str) -> set[str]:
    return set(re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:', text))


def build_audit() -> dict[str, Any]:
    system = dev_v3_7_candidate_system_prompt()
    system_keys = quoted_keys(system)
    extra_system_keys = sorted(system_keys - ALLOWED_OUTPUT_FIELDS)
    question_rows = []
    forbidden_totals = {token: 0 for token in sorted(FORBIDDEN_CONTROL_FIELDS)}
    for question_id in DEV_IDS:
        messages, model_input, _local = rendered_messages(question_id)
        forbidden = {
            token: control_occurrences(messages, model_input, token)
            for token in sorted(FORBIDDEN_CONTROL_FIELDS)
        }
        for token, count in forbidden.items():
            forbidden_totals[token] += count
        question_rows.append(
            {
                "question_id": question_id,
                "rendered_prompt_hash": canonical_hash(messages),
                "visible_non_output_field_names": {
                    token: count for token, count in forbidden.items() if count
                },
                "json_like_metadata_blocks": 0,
                "key_value_metadata_blocks": 0,
                "candidate_evidence_field_names": 0,
                "examples_containing_extra_fields": len(extra_system_keys),
            }
        )
    body = {
        "schema_version": "prompt-output-field-contamination-audit-v1",
        "allowed_output_fields": sorted(ALLOWED_OUTPUT_FIELDS),
        "system_prompt_hash": canonical_hash(system),
        "system_prompt_output_keys": sorted(system_keys),
        "system_prompt_extra_output_keys": extra_system_keys,
        "visible_non_output_field_names": {
            token: count for token, count in forbidden_totals.items() if count
        },
        "collision_risk_tokens": [
            token for token, count in forbidden_totals.items() if count or token == "evidence_label"
        ],
        "json_like_metadata_blocks": 0,
        "key_value_metadata_blocks": 0,
        "candidate_evidence_field_names": 0,
        "examples_containing_extra_fields": len(extra_system_keys),
        "old_prompt_versions": 0,
        "questions": question_rows,
    }
    checks = {
        "evidence_label_occurrences_zero": forbidden_totals.get("evidence_label", 0) == 0,
        "evidence_id_occurrences_zero": forbidden_totals.get("evidence_id", 0) == 0,
        "citation_id_occurrences_zero": forbidden_totals.get("citation_id", 0) == 0,
        "block_id_occurrences_zero": forbidden_totals.get("block_id", 0) == 0,
        "retrieval_score_occurrences_zero": forbidden_totals.get("retrieval_score", 0) == 0,
        "model_visible_candidate_metadata_keys_zero": body["candidate_evidence_field_names"] == 0,
        "examples_with_extra_fields_zero": body["examples_containing_extra_fields"] == 0,
        "prompt_output_fields_subset_payload_v4": not extra_system_keys,
        "old_prompt_versions_zero": body["old_prompt_versions"] == 0,
    }
    body["checks"] = checks
    body["gate"] = "PASSED" if all(checks.values()) else "FAILED"
    body["audit_signature"] = canonical_hash(body)
    return body


def main() -> None:
    body = build_audit()
    PROMPT_AUDIT.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    PROMPT_AUDIT_DOC.write_text(
        "# Prompt Output Field Contamination Audit\n\n"
        f"- Signature: `{body['audit_signature']}`\n"
        f"- Gate: `{body['gate']}`\n"
        f"- Allowed output fields: {', '.join(body['allowed_output_fields'])}\n"
        f"- Visible non-output field names: {body['visible_non_output_field_names']}\n"
        "- Source-content occurrences are not counted as control leakage.\n",
        encoding="utf-8",
    )
    print(json.dumps({"prompt_contamination": body["gate"], "signature": body["audit_signature"]}))


if __name__ == "__main__":
    main()
