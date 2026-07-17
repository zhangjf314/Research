"""Evidence Presentation v2 candidate rendering and audits."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from paper_research.generation.schema_reliability import (
    DEV_V3_7_CANDIDATE_PROMPT_VERSION,
    LOCAL_ENVELOPE_V4_VERSION,
    MODEL_PAYLOAD_V4_VERSION,
    PAYLOAD_V4_ADAPTER,
    LocalEnvelopeV4,
    dev_v3_7_candidate_system_prompt,
)

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DOCS, canonical_hash
    from scripts.evidence_qa_dev_v3_3_lib import output_budget, safe_model_input
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DATA, DOCS, canonical_hash  # type: ignore[no-redef]
    from evidence_qa_dev_v3_3_lib import output_budget, safe_model_input  # type: ignore[no-redef]

PRESENTATION_VERSION = "evidence-presentation-v2-candidate"
SELECTED_FORMAT = "uniform-unnumbered-delimiter"
BEGIN = "--- BEGIN PASSAGE ---"
END = "--- END PASSAGE ---"

PROTOCOL = DATA / "evidence-presentation-v2-protocol.json"
PROTOCOL_DOC = DOCS / "evidence-presentation-v2-protocol.md"
MAPPING_AUDIT = DATA / "evidence-presentation-v2-local-mapping-audit-v1.json"
METADATA_AUDIT = DATA / "dev-v3-5-model-visible-metadata-audit-v1.json"
METADATA_AUDIT_DOC = DOCS / "dev-v3-5-model-visible-metadata-audit-v1.md"
PROMPT_AUDIT = DATA / "prompt-output-field-contamination-audit-v1.json"
PROMPT_AUDIT_DOC = DOCS / "prompt-output-field-contamination-audit-v1.md"
RENDER_PREFLIGHT = DATA / "dev-v3-6-prompt-rendering-preflight-v1.json"
RENDER_PREFLIGHT_DOC = DOCS / "dev-v3-6-prompt-rendering-preflight-v1.md"
COPY_ANALYSIS = DATA / "dev-v3-5-output-field-copy-analysis-v1.json"
COPY_ANALYSIS_DOC = DOCS / "dev-v3-5-output-field-copy-analysis-v1.md"
FORENSICS = DATA / "dev-v3-5-evidence-label-forensics-v1.json"
FORENSICS_DOC = DOCS / "dev-v3-5-evidence-label-forensics-v1.md"
READINESS = DATA / "evidence-presentation-v2-readiness-v1.json"
READINESS_DOC = DOCS / "evidence-presentation-v2-readiness-v1.md"

ALLOWED_OUTPUT_FIELDS = {
    "answerable",
    "required_claim_results",
    "required_claim_id",
    "claim_text",
    "omission_reason",
    "refusal_reason",
}

FORBIDDEN_CONTROL_FIELDS = {
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
    "evidence_role",
    "original",
    "adjacent",
    "human_label",
    "gold",
}

AUDITED_TOKENS = sorted(
    FORBIDDEN_CONTROL_FIELDS | {"label", "Evidence A", "Evidence B", "Evidence C"}
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_passages(full: dict[str, Any]) -> list[str]:
    passages = []
    for claim in full["required_claims"]:
        for allocated in claim["allocated_evidence"]:
            passages.append(allocated["summary"])
    return passages


def presentation_input(question_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    _old_safe, full, registry, trace = safe_model_input(question_id)
    claims = []
    local_mapping = {}
    for claim in full["required_claims"]:
        passages = []
        local_rows = []
        for position, allocated in enumerate(claim["allocated_evidence"], start=1):
            text = allocated["summary"]
            passages.append(text)
            local_rows.append(
                {
                    "passage_position": position,
                    "evidence_id": allocated["evidence_id"],
                    "citation_ids": allocated["citation_ids"],
                    "text_hash": sha256_text(text),
                    "text": text,
                    "source_metadata": {
                        key: value
                        for key, value in allocated.items()
                        if key not in {"summary", "citation_ids"}
                    },
                }
            )
        claims.append(
            {
                "required_claim_id": claim["required_claim_id"],
                "required_claim_text": claim["required_claim_text"],
                "passages": passages,
                "omission_policy": claim["omission_policy"],
            }
        )
        local_mapping[claim["required_claim_id"]] = local_rows
    model_input = {
        "question": full["question"],
        "answerability_expectation": full["answerability_expectation"],
        "required_claims": claims,
        "output_budget": output_budget(len(claims)),
    }
    local = {
        "question_id": question_id,
        "citation_registry": registry.model_dump(mode="json"),
        "citation_registry_hash": registry.registry_hash,
        "local_mapping": local_mapping,
        "trace_hash": canonical_hash(trace),
    }
    return model_input, local


def render_user_prompt(model_input: dict[str, Any]) -> str:
    lines = [
        "Task input",
        "",
        "Question",
        model_input["question"],
        "",
        "Answerability expectation",
        str(model_input["answerability_expectation"]).lower(),
        "",
    ]
    for claim in model_input["required_claims"]:
        lines.extend(
            [
                "Required claim",
                f"required_claim_id {claim['required_claim_id']}",
                "Required content",
                claim["required_claim_text"],
                "Passages",
            ]
        )
        for passage in claim["passages"]:
            lines.extend([BEGIN, passage, END])
        lines.extend(
            [
                "Omission instruction",
                "Use omission_reason for this required_claim_id when the passages "
                "do not establish the required content.",
                "",
            ]
        )
    lines.append(json.dumps({"output_budget": model_input["output_budget"]}, ensure_ascii=False))
    return "\n".join(lines)


def rendered_messages(
    question_id: str,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    model_input, local = presentation_input(question_id)
    return [
        {"role": "system", "content": dev_v3_7_candidate_system_prompt()},
        {"role": "user", "content": render_user_prompt(model_input)},
    ], model_input, local


def token_occurrences(text: str, token: str) -> int:
    if " " in token:
        return text.count(token)
    return len(re.findall(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", text))


def source_content_occurrences(model_input: dict[str, Any], token: str) -> int:
    passage_hits = sum(
        token_occurrences(passage, token)
        for claim in model_input["required_claims"]
        for passage in claim["passages"]
    )
    claim_hits = sum(
        token_occurrences(claim["required_claim_text"], token)
        for claim in model_input["required_claims"]
    )
    question_hits = token_occurrences(model_input["question"], token)
    return passage_hits + claim_hits + question_hits


def control_occurrences(
    messages: list[dict[str, str]], model_input: dict[str, Any], token: str
) -> int:
    total = sum(token_occurrences(message["content"], token) for message in messages)
    return max(0, total - source_content_occurrences(model_input, token))


def passage_hashes(model_input: dict[str, Any]) -> list[str]:
    return [
        sha256_text(passage)
        for claim in model_input["required_claims"]
        for passage in claim["passages"]
    ]


def build_protocol() -> dict[str, Any]:
    candidates = {
        "uniform_unnumbered_delimiter": {
            "copyable_field_risk": "low",
            "boundary_clarity": "high",
            "token_overhead": "medium",
            "auditability": "high",
            "local_mapping": "position_only",
            "requires_model_label_output": False,
            "selected": True,
        },
        "plain_paragraph_blank_lines": {
            "copyable_field_risk": "low",
            "boundary_clarity": "medium",
            "token_overhead": "low",
            "auditability": "medium",
            "local_mapping": "position_only",
            "requires_model_label_output": False,
            "selected": False,
        },
        "xml_passage_tags": {
            "copyable_field_risk": "medium",
            "boundary_clarity": "high",
            "token_overhead": "medium",
            "auditability": "high",
            "local_mapping": "position_only",
            "requires_model_label_output": False,
            "selected": False,
        },
    }
    body = {
        "schema_version": "evidence-presentation-v2-protocol-v1",
        "version": PRESENTATION_VERSION,
        "selected_format": SELECTED_FORMAT,
        "principles": [
            "Separate control metadata from semantic passage text.",
            "The model sees passage text and minimal reading boundaries only.",
            "The model does not see copyable candidate evidence field names.",
            "The local CitationRegistry keeps full mapping metadata.",
            "The model never outputs evidence identifiers.",
        ],
        "model_visible_candidate_metadata": [],
        "retained_local_metadata": [
            "passage_position",
            "citation_registry_id",
            "evidence_id",
            "paper_id",
            "page",
            "block_id",
            "original_or_adjacent",
            "retrieval_score",
            "source_hash",
        ],
        "format_candidates": candidates,
        "payload_schema_version": MODEL_PAYLOAD_V4_VERSION,
        "payload_schema_hash": canonical_hash(PAYLOAD_V4_ADAPTER.json_schema()),
        "local_envelope_version": LOCAL_ENVELOPE_V4_VERSION,
        "local_envelope_hash": canonical_hash(LocalEnvelopeV4.model_json_schema()),
        "prompt_version": DEV_V3_7_CANDIDATE_PROMPT_VERSION,
        "prompt_hash": canonical_hash(dev_v3_7_candidate_system_prompt()),
        "next_live_authorized": False,
    }
    body["protocol_signature"] = canonical_hash(body)
    return body
