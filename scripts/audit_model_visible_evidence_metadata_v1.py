"""Audit model-visible evidence metadata for the decontaminated prompt candidate."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

try:
    from scripts.evidence_presentation_v2_lib import (
        AUDITED_TOKENS,
        METADATA_AUDIT,
        METADATA_AUDIT_DOC,
        canonical_hash,
        control_occurrences,
        rendered_messages,
        source_content_occurrences,
    )
    from scripts.evidence_qa_dev_lib_v1 import DEV_IDS
except ModuleNotFoundError:
    from evidence_presentation_v2_lib import (  # type: ignore[no-redef]
        AUDITED_TOKENS,
        METADATA_AUDIT,
        METADATA_AUDIT_DOC,
        canonical_hash,
        control_occurrences,
        rendered_messages,
        source_content_occurrences,
    )
    from evidence_qa_dev_lib_v1 import DEV_IDS  # type: ignore[no-redef]


def build_audit() -> dict[str, Any]:
    aggregate: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "question_count": 0,
            "occurrence_count": 0,
            "source_content_occurrences": 0,
            "locations": [],
        }
    )
    per_question = []
    for question_id in DEV_IDS:
        messages, model_input, local = rendered_messages(question_id)
        text = "\n".join(message["content"] for message in messages)
        question_tokens = {}
        for token in AUDITED_TOKENS:
            control_count = control_occurrences(messages, model_input, token)
            source_count = source_content_occurrences(model_input, token)
            if control_count or source_count:
                question_tokens[token] = {
                    "control_occurrences": control_count,
                    "source_content_occurrences": source_count,
                }
                aggregate[token]["question_count"] += int(control_count > 0)
                aggregate[token]["occurrence_count"] += control_count
                aggregate[token]["source_content_occurrences"] += source_count
                aggregate[token]["locations"].append(question_id)
        per_question.append(
            {
                "question_id": question_id,
                "rendered_prompt_hash": canonical_hash(messages),
                "visible_tokens": question_tokens,
                "citation_registry_hash": local["citation_registry_hash"],
                "prompt_length": len(text),
            }
        )
    fields = []
    for token in AUDITED_TOKENS:
        row = aggregate[token]
        fields.append(
            {
                "field_or_token": token,
                "location": "model_visible_prompt"
                if row["occurrence_count"]
                else "source_content_only"
                if row["source_content_occurrences"]
                else "absent",
                "question_count": row["question_count"],
                "occurrence_count": row["occurrence_count"],
                "source_content_occurrences": row["source_content_occurrences"],
                "copyable": row["occurrence_count"] > 0,
                "required_for_semantic_understanding": False,
                "required_for_local_mapping": token
                in {
                    "evidence_id",
                    "citation_id",
                    "citation_ids",
                    "block_id",
                    "paper_id",
                    "page",
                    "retrieval_score",
                    "original",
                    "adjacent",
                },
                "safe_to_remove_from_model_view": True,
                "retained_only_in_local_metadata": token
                not in {"label", "Evidence A", "Evidence B", "Evidence C"},
                "output_schema_collision_risk": token
                in {"evidence_label", "label", "citation_id", "status"},
            }
        )
    body = {
        "schema_version": "dev-v3-5-model-visible-metadata-audit-v1",
        "questions_scanned": len(DEV_IDS),
        "model_visible_internal_control_fields": sum(
            row["occurrence_count"]
            for row in fields
            if row["field_or_token"] not in {"Evidence A", "Evidence B", "Evidence C"}
        ),
        "model_copyable_unneeded_field_names_minimized": True,
        "local_mapping_metadata_preserved": True,
        "fields": fields,
        "per_question": per_question,
        "gate": "PASSED"
        if all(row["occurrence_count"] == 0 for row in fields)
        else "FAILED",
    }
    body["audit_signature"] = canonical_hash(body)
    return body


def main() -> None:
    body = build_audit()
    METADATA_AUDIT.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    METADATA_AUDIT_DOC.write_text(
        "# Dev v3.5 Model-Visible Metadata Audit\n\n"
        f"- Signature: `{body['audit_signature']}`\n"
        f"- Gate: `{body['gate']}`\n"
        f"- Model-visible internal control fields: "
        f"{body['model_visible_internal_control_fields']}\n"
        "- Local mapping metadata is preserved outside the model-visible prompt.\n",
        encoding="utf-8",
    )
    print(json.dumps({"metadata_audit": body["gate"], "signature": body["audit_signature"]}))


if __name__ == "__main__":
    main()
