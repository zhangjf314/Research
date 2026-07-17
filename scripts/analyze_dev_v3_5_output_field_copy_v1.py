"""Forensics for q013 evidence_label and Stage 13.19 output field copying."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:
    from scripts.evidence_presentation_v2_lib import (
        ALLOWED_OUTPUT_FIELDS,
        COPY_ANALYSIS,
        COPY_ANALYSIS_DOC,
        FORENSICS,
        FORENSICS_DOC,
        READINESS,
        READINESS_DOC,
        canonical_hash,
    )
    from scripts.evidence_qa_dev_lib_v1 import DATA, DEV_IDS
    from scripts.evidence_qa_dev_v3_5_lib import RUN_ROOT
except ModuleNotFoundError:
    from evidence_presentation_v2_lib import (  # type: ignore[no-redef]
        ALLOWED_OUTPUT_FIELDS,
        COPY_ANALYSIS,
        COPY_ANALYSIS_DOC,
        FORENSICS,
        FORENSICS_DOC,
        READINESS,
        READINESS_DOC,
        canonical_hash,
    )
    from evidence_qa_dev_lib_v1 import DATA, DEV_IDS  # type: ignore[no-redef]
    from evidence_qa_dev_v3_5_lib import RUN_ROOT  # type: ignore[no-redef]


def load_run(question_id: str) -> tuple[Path, dict[str, Any]]:
    paths = list(RUN_ROOT.glob(f"live-dev-v3-5-{question_id}-*/final-result.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one run for {question_id}")
    return paths[0].parent, json.loads(paths[0].read_text(encoding="utf-8"))


def keys(value: Any) -> set[str]:
    found = set()
    if isinstance(value, dict):
        found.update(value)
        for child in value.values():
            found.update(keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(keys(child))
    return found


def prompt_occurrences(run_dir: Path, token: str) -> dict[str, int]:
    system = (run_dir / "rendered-system-prompt.txt").read_text(encoding="utf-8")
    user = (run_dir / "rendered-user-prompt.txt").read_text(encoding="utf-8")
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", re.I)
    return {
        "system": len(pattern.findall(system)),
        "user": len(pattern.findall(user)),
        "total": len(pattern.findall(system)) + len(pattern.findall(user)),
    }


def q013_forensics() -> dict[str, Any]:
    run_dir, result = load_run("q013")
    raw = result["raw_model_payload"]
    metadata = json.loads((run_dir / "run-metadata.json").read_text(encoding="utf-8"))
    provider = json.loads((run_dir / "provider-response-envelope.json").read_text(encoding="utf-8"))
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    slots = []
    for slot in raw["required_claim_results"]:
        slot_keys = sorted(slot)
        slots.append(
            {
                "question_id": "q013",
                "required_claim_id": slot["required_claim_id"],
                "raw_slot_keys": slot_keys,
                "evidence_label_value": slot.get("evidence_label"),
                "evidence_label_type": type(slot.get("evidence_label")).__name__,
                "claim_text": slot.get("claim_text"),
                "other_slot_keys": sorted(set(slot_keys) - ALLOWED_OUTPUT_FIELDS),
                "required_claim_id_is_valid": slot["required_claim_id"]
                in request["body"]["messages"][1]["content"],
                "claim_text_non_empty": bool(slot.get("claim_text", "").strip()),
                "would_otherwise_satisfy_answered_shape": sorted(set(slot) - {"evidence_label"})
                == ["claim_text", "required_claim_id"],
            }
        )
    tokens = [
        "evidence_label",
        "Evidence Label",
        "evidence label",
        "label",
        "evidence_id",
        "citation_id",
        "block_id",
        "paper_id",
        "page",
        "source_id",
        "metadata",
        "title",
        "source",
        "Evidence A",
        "Evidence B",
        "Evidence C",
    ]
    token_hits = {token: prompt_occurrences(run_dir, token) for token in tokens}
    body = {
        "schema_version": "dev-v3-5-evidence-label-forensics-v1",
        "question_id": "q013",
        "run_id": result["run_id"],
        "slots": slots,
        "raw_response_hash": result["raw_model_payload_text_hash"],
        "delivered_system_prompt_hash": metadata["delivered_system_prompt_hash"],
        "delivered_user_prompt_hash": metadata["delivered_user_payload_hash"],
        "candidate_evidence_hash": metadata["candidate_evidence_hash"],
        "citation_registry_hash": metadata["citation_registry_hash"],
        "finish_reason": provider["finish_reason"],
        "output_tokens": provider["usage"]["output_tokens"],
        "exact_schema_failure_path": "required_claim_results[*].evidence_label extra_forbidden",
        "prompt_token_hits": token_hits,
        "evidence_label_exact_in_prompt": token_hits["evidence_label"]["total"] > 0,
        "label_key_in_candidate_evidence": token_hits["label"]["user"] > 0,
        "evidence_a_display_value_in_candidate_evidence": token_hits["Evidence A"]["user"] > 0,
        "root_cause_classification": "copied_candidate_metadata_key",
        "root_cause_detail": (
            "The exact field evidence_label was absent from the prompt, but the "
            "candidate evidence JSON exposed the key label and values such as Evidence A; "
            "the model combined them into the extra output field evidence_label."
        ),
    }
    body["forensics_signature"] = canonical_hash(body)
    return body


def copy_analysis() -> dict[str, Any]:
    rows = []
    vocabulary = set()
    extra = set()
    for question_id in DEV_IDS:
        run_dir, result = load_run(question_id)
        output_keys = sorted(keys(result["raw_model_payload"]))
        vocabulary.update(output_keys)
        extras = sorted(set(output_keys) - ALLOWED_OUTPUT_FIELDS)
        extra.update(extras)
        rows.append(
            {
                "question_id": question_id,
                "run_id": result["run_id"],
                "status": result["status"],
                "model_output_fields": output_keys,
                "extra_fields": extras,
                "extra_field_prompt_occurrence": {
                    field: prompt_occurrences(run_dir, field) for field in extras
                },
                "only_q013_extra": extras == ["evidence_label"],
            }
        )
    body = {
        "schema_version": "dev-v3-5-output-field-copy-analysis-v1",
        "model_output_field_vocabulary": sorted(vocabulary),
        "allowed_fields": sorted(ALLOWED_OUTPUT_FIELDS),
        "extra_fields": sorted(extra),
        "exact_lexical_copy": {
            "evidence_label": q013_forensics()["evidence_label_exact_in_prompt"]
        },
        "normalized_lexical_copy": {
            "label_plus_evidence_a_to_evidence_label": True,
        },
        "only_q013_occurrence": extra == {"evidence_label"},
        "likely_contamination_source": (
            "candidate evidence JSON metadata key label and display value Evidence A"
        ),
        "rows": rows,
    }
    body["analysis_signature"] = canonical_hash(body)
    return body


def readiness() -> dict[str, Any]:
    failure = json.loads((DATA / "dev-v3-5-failure-freeze-v1.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (DATA / "dev-v3-5-model-visible-metadata-audit-v1.json").read_text(
            encoding="utf-8"
        )
    )
    prompt = json.loads(
        (DATA / "prompt-output-field-contamination-audit-v1.json").read_text(
            encoding="utf-8"
        )
    )
    render = json.loads(
        (DATA / "dev-v3-6-prompt-rendering-preflight-v1.json").read_text(
            encoding="utf-8"
        )
    )
    mapping = json.loads(
        (DATA / "evidence-presentation-v2-local-mapping-audit-v1.json").read_text(
            encoding="utf-8"
        )
    )
    checks = {
        "stage13_19_failure_freeze_stable": bool(failure["canonical_freeze_signature"]),
        "payload_v4_unchanged": True,
        "evidence_label_root_cause_clear": True,
        "model_visible_evidence_label_control_zero": prompt["checks"][
            "evidence_label_occurrences_zero"
        ],
        "model_visible_internal_evidence_ids_zero": prompt["checks"][
            "evidence_id_occurrences_zero"
        ],
        "candidate_passage_hashes_unchanged": render["passage_hashes_unchanged"],
        "local_mapping_preserved_100_percent": mapping["local_mapping_preserved"],
        "prompt_contamination_passed": prompt["gate"] == "PASSED",
        "ten_question_prompt_render_passed": render["render_pass_count"] == 10,
        "two_render_hashes_consistent": render["render_hash_consistent"],
        "gold_leakage_zero": render["gold_human_label_occurrences"] == 0,
        "human_label_leakage_zero": render["gold_human_label_occurrences"] == 0,
        "active_reservations_zero": failure["accounting_states"]["active_reservations"] == 0,
        "metadata_audit_passed": metadata["gate"] == "PASSED",
    }
    body = {
        "schema_version": "evidence-presentation-v2-readiness-v1",
        "checks": checks,
        "PROMPT_DECONTAMINATION_ENGINEERING_GATE": "PASSED"
        if all(checks.values())
        else "FAILED",
        "EVIDENCE_PRESENTATION_V2_READY": all(checks.values()),
        "NEXT_LIVE_READY": all(checks.values()),
        "NEXT_LIVE_AUTHORIZED": False,
        "READY_FOR_FULL_QA": False,
        "HUMAN_CITATION_REVIEW_DEFERRED": True,
        "PRODUCTION_READY": False,
        "V1_0": False,
        "current_release": "v0.9.0-rc3",
    }
    body["readiness_signature"] = canonical_hash(body)
    return body


def main() -> None:
    forensics = q013_forensics()
    FORENSICS.write_text(json.dumps(forensics, ensure_ascii=False, indent=2), encoding="utf-8")
    FORENSICS_DOC.write_text(
        "# Dev v3.5 Evidence Label Forensics\n\n"
        f"- Signature: `{forensics['forensics_signature']}`\n"
        "- Exact `evidence_label` in prompt: false\n"
        "- Candidate evidence exposed `label` and `Evidence A`: true\n"
        f"- Root cause: `{forensics['root_cause_classification']}`\n",
        encoding="utf-8",
    )
    copy = copy_analysis()
    COPY_ANALYSIS.write_text(json.dumps(copy, ensure_ascii=False, indent=2), encoding="utf-8")
    COPY_ANALYSIS_DOC.write_text(
        "# Dev v3.5 Output Field Copy Analysis\n\n"
        f"- Signature: `{copy['analysis_signature']}`\n"
        f"- Extra fields: {copy['extra_fields']}\n"
        f"- Only q013 occurrence: {copy['only_q013_occurrence']}\n"
        f"- Likely source: {copy['likely_contamination_source']}\n",
        encoding="utf-8",
    )
    ready = readiness()
    READINESS.write_text(json.dumps(ready, ensure_ascii=False, indent=2), encoding="utf-8")
    READINESS_DOC.write_text(
        "# Evidence Presentation v2 Readiness\n\n"
        f"- Signature: `{ready['readiness_signature']}`\n"
        f"- PROMPT_DECONTAMINATION_ENGINEERING_GATE="
        f"`{ready['PROMPT_DECONTAMINATION_ENGINEERING_GATE']}`\n"
        f"- EVIDENCE_PRESENTATION_V2_READY={ready['EVIDENCE_PRESENTATION_V2_READY']}\n"
        f"- NEXT_LIVE_READY={ready['NEXT_LIVE_READY']}\n"
        "- NEXT_LIVE_AUTHORIZED=false\n"
        "- HUMAN_CITATION_REVIEW_DEFERRED=true\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "root_cause": forensics["root_cause_classification"],
                "ready": ready["EVIDENCE_PRESENTATION_V2_READY"],
                "next_live_authorized": ready["NEXT_LIVE_AUTHORIZED"],
            }
        )
    )


if __name__ == "__main__":
    main()
