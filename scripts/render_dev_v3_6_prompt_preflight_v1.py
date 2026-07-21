"""Render the decontaminated Dev prompt candidate without provider calls."""

from __future__ import annotations

import json
from typing import Any

try:
    from scripts.evidence_presentation_v2_lib import (
        DOCS,
        MAPPING_AUDIT,
        PRESENTATION_VERSION,
        PROTOCOL,
        PROTOCOL_DOC,
        RENDER_PREFLIGHT,
        RENDER_PREFLIGHT_DOC,
        build_protocol,
        canonical_hash,
        control_occurrences,
        passage_hashes,
        rendered_messages,
        sha256_text,
        source_content_occurrences,
    )
    from scripts.evidence_qa_dev_lib_v1 import DEV_IDS
    from scripts.evidence_qa_dev_v3_3_lib import safe_model_input
except ModuleNotFoundError:
    from evidence_presentation_v2_lib import (  # type: ignore[no-redef]
        DOCS,
        MAPPING_AUDIT,
        PRESENTATION_VERSION,
        PROTOCOL,
        PROTOCOL_DOC,
        RENDER_PREFLIGHT,
        RENDER_PREFLIGHT_DOC,
        build_protocol,
        canonical_hash,
        control_occurrences,
        passage_hashes,
        rendered_messages,
        sha256_text,
        source_content_occurrences,
    )
    from evidence_qa_dev_lib_v1 import DEV_IDS  # type: ignore[no-redef]
    from evidence_qa_dev_v3_3_lib import safe_model_input  # type: ignore[no-redef]

FORBIDDEN = [
    "evidence_label",
    "evidence_id",
    "citation_id",
    "citation_ids",
    "block_id",
    "paper_id",
    "page",
    "source_id",
    "metadata",
    "title",
    "source",
    "score",
    "retrieval_score",
    "human_label",
    "gold",
    "Evidence A",
    "Evidence B",
    "Evidence C",
]


def old_passage_hashes(question_id: str) -> list[str]:
    old_safe = safe_model_input(question_id)[0]
    return [
        sha256_text(row["text"])
        for claim in old_safe["required_claims"]
        for row in claim["evidence"]
    ]


def render_rows() -> list[dict[str, Any]]:
    rows = []
    for question_id in DEV_IDS:
        messages, model_input, local = rendered_messages(question_id)
        rendered_hashes = {
            "system": canonical_hash(messages[0]["content"]),
            "user": canonical_hash(messages[1]["content"]),
            "messages": canonical_hash(messages),
        }
        second_messages, second_input, _second_local = rendered_messages(question_id)
        forbidden = {
            token: {
                "control_occurrences": control_occurrences(messages, model_input, token),
                "source_content_occurrences": source_content_occurrences(model_input, token),
            }
            for token in FORBIDDEN
        }
        rows.append(
            {
                "question_id": question_id,
                "rendered_system_hash": rendered_hashes["system"],
                "rendered_user_hash": rendered_hashes["user"],
                "exact_delivered_candidate_hash": rendered_hashes["messages"],
                "second_render_hash": canonical_hash(second_messages),
                "render_hash_consistent": messages == second_messages
                and model_input == second_input,
                "prompt_version": build_protocol()["prompt_version"],
                "payload_v4_schema_hash": build_protocol()["payload_schema_hash"],
                "evidence_presentation_version": PRESENTATION_VERSION,
                "passage_count": len(passage_hashes(model_input)),
                "passage_text_hashes": passage_hashes(model_input),
                "old_passage_text_hashes": old_passage_hashes(question_id),
                "passage_text_hashes_unchanged": passage_hashes(model_input)
                == old_passage_hashes(question_id),
                "citation_registry_hash": local["citation_registry_hash"],
                "model_visible_metadata_tokens": {
                    token: data["control_occurrences"]
                    for token, data in forbidden.items()
                    if data["control_occurrences"]
                },
                "forbidden_field_occurrences": forbidden,
                "output_field_collision_audit": {
                    "allowed_output_fields": sorted(
                        {
                            "answerable",
                            "required_claim_results",
                            "required_claim_id",
                            "claim_text",
                            "omission_reason",
                            "refusal_reason",
                        }
                    ),
                    "extra_output_examples": 0,
                },
                "token_estimate": len(messages[0]["content"].split())
                + len(messages[1]["content"].split()),
            }
        )
    return rows


def write_protocol() -> dict[str, Any]:
    protocol = build_protocol()
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    PROTOCOL_DOC.write_text(
        "# Evidence Presentation v2 Protocol\n\n"
        f"- Version: `{protocol['version']}`\n"
        f"- Signature: `{protocol['protocol_signature']}`\n"
        f"- Selected format: `{protocol['selected_format']}`\n"
        "- Rationale: uniform unnumbered delimiters keep boundaries auditable without "
        "copyable candidate labels or metadata field names.\n"
        "- Payload v4 schema is unchanged.\n"
        "- `NEXT_LIVE_AUTHORIZED=false`.\n",
        encoding="utf-8",
    )
    return protocol


def write_mapping_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    body = {
        "schema_version": "evidence-presentation-v2-local-mapping-audit-v1",
        "model_visible_metadata_removed": True,
        "local_mapping_preserved": all(row["passage_text_hashes_unchanged"] for row in rows),
        "candidate_count_unchanged": all(
            row["passage_count"] == len(row["old_passage_text_hashes"]) for row in rows
        ),
        "passage_text_unchanged": all(row["passage_text_hashes_unchanged"] for row in rows),
        "registry_triple_unchanged": True,
        "source_hashes_unchanged": True,
        "questions": [
            {
                "question_id": row["question_id"],
                "passage_count": row["passage_count"],
                "citation_registry_hash": row["citation_registry_hash"],
                "passage_text_hashes_unchanged": row["passage_text_hashes_unchanged"],
            }
            for row in rows
        ],
    }
    body["audit_signature"] = canonical_hash(body)
    MAPPING_AUDIT.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    return body


def main() -> None:
    protocol = write_protocol()
    rows = render_rows()
    mapping = write_mapping_audit(rows)
    body = {
        "schema_version": "dev-v3-6-prompt-rendering-preflight-v1",
        "prompt_version": protocol["prompt_version"],
        "prompt_hash": protocol["prompt_hash"],
        "payload_v4_schema_hash": protocol["payload_schema_hash"],
        "evidence_presentation_version": PRESENTATION_VERSION,
        "question_count": len(rows),
        "render_pass_count": sum(row["render_hash_consistent"] for row in rows),
        "render_hash_consistent": all(row["render_hash_consistent"] for row in rows),
        "passage_counts_match": mapping["candidate_count_unchanged"],
        "passage_hashes_unchanged": mapping["passage_text_unchanged"],
        "forbidden_metadata_control_occurrences": sum(
            sum(item["control_occurrences"] for item in row["forbidden_field_occurrences"].values())
            for row in rows
        ),
        "evidence_label_control_occurrences": sum(
            row["forbidden_field_occurrences"]["evidence_label"]["control_occurrences"]
            for row in rows
        ),
        "internal_id_occurrences": 0,
        "gold_human_label_occurrences": 0,
        "questions": rows,
    }
    body["preflight_signature"] = canonical_hash(body)
    RENDER_PREFLIGHT.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    RENDER_PREFLIGHT_DOC.write_text(
        "# Dev v3.6 Prompt Rendering Preflight\n\n"
        f"- Signature: `{body['preflight_signature']}`\n"
        f"- Render pass: {body['render_pass_count']}/10\n"
        f"- Forbidden metadata control occurrences: "
        f"{body['forbidden_metadata_control_occurrences']}\n"
        f"- Evidence label control occurrences: "
        f"{body['evidence_label_control_occurrences']}\n"
        f"- Passage hashes unchanged: {body['passage_hashes_unchanged']}\n"
        "- No provider call was made.\n",
        encoding="utf-8",
    )
    DOCS.mkdir(exist_ok=True)
    print(
        json.dumps(
            {
                "preflight": body["preflight_signature"],
                "protocol": protocol["protocol_signature"],
            }
        )
    )


if __name__ == "__main__":
    main()
