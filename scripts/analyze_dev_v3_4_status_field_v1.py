"""Deterministic forensics for Stage 13.16 slot status failures."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from paper_research.generation.required_claim_output import RequiredClaimValidationError
from paper_research.generation.schema_reliability import derive_slot_status_v1

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"
DOCS = ROOT / "docs"
RUN_ROOT = DATA / "evidence-qa-dev-v3-4/runs"
OUTPUT = DATA / "dev-v3-4-status-field-forensics-v1.json"
OUTPUT_DOC = DOCS / "dev-v3-4-status-field-forensics-v1.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_slot(
    *,
    question_id: str,
    slot: dict[str, Any],
    valid_ids: set[str],
    answerable: bool,
    refusal_present: bool,
    refusal_value: Any,
    finish_reason: str | None,
    output_tokens: int,
) -> dict[str, Any]:
    expected_fields = {
        "required_claim_id",
        "status",
        "claim_text",
        "omission_reason",
    }
    extra = sorted(set(slot) - expected_fields)
    missing = sorted(expected_fields - set(slot))
    raw_status = slot.get("status")
    claim = slot.get("claim_text")
    omission = slot.get("omission_reason")
    without_status = {key: value for key, value in slot.items() if key != "status"}
    shape_valid = True
    shape_failure = None
    try:
        derive_slot_status_v1(without_status)
    except RequiredClaimValidationError as exc:
        shape_valid = False
        shape_failure = {"code": exc.code, "message": str(exc)}
    other_fields_structurally_valid = (
        not extra
        and not missing
        and isinstance(slot.get("required_claim_id"), str)
        and slot.get("required_claim_id") in valid_ids
        and (claim is None or isinstance(claim, str))
        and (omission is None or isinstance(omission, str))
    )
    paths = []
    if raw_status not in {"answered", "unsupported", "not_applicable"}:
        paths.append("required_claim_results[].status")
    if not refusal_present:
        paths.append("refusal_reason")
    if not shape_valid:
        paths.append("required_claim_results[].claim_text/omission_reason")
    return {
        "question_id": question_id,
        "required_claim_id": slot.get("required_claim_id"),
        "raw_status_value": raw_status,
        "raw_status_type": type(raw_status).__name__,
        "claim_text_presence": "claim_text" in slot,
        "claim_text_length": len(claim) if isinstance(claim, str) else None,
        "omission_reason_value": omission,
        "omission_reason_type": type(omission).__name__,
        "answerable": answerable,
        "top_level_refusal_reason_presence": refusal_present,
        "top_level_refusal_reason_value": refusal_value,
        "other_slot_fields_structurally_valid": other_fields_structurally_valid,
        "content_shape_v3_valid_without_status": shape_valid,
        "content_shape_failure": shape_failure,
        "required_claim_id_valid": slot.get("required_claim_id") in valid_ids,
        "extra_fields": extra,
        "missing_fields": missing,
        "prompt_legal_status_values": ["answered", "unsupported", "not_applicable"],
        "prompt_example_status_values": [],
        "raw_finish_reason": finish_reason,
        "output_token_count": output_tokens,
        "exact_validation_failure_paths": paths,
    }


def build() -> dict[str, Any]:
    rows = []
    question_rows = []
    statuses: Counter[str] = Counter()
    for run_dir in sorted(RUN_ROOT.glob("live-dev-v3-4-*")):
        raw = read_json(run_dir / "raw-model-payload.json")
        required = read_json(run_dir / "required-claims-input.json")
        envelope = read_json(run_dir / "provider-response-envelope.json")
        valid_ids = {
            row["required_claim_id"] for row in required["required_claims"]
        }
        refusal_present = "refusal_reason" in raw
        question_slots = []
        for slot in raw.get("required_claim_results", []):
            status = slot.get("status", "<missing>")
            statuses[str(status)] += 1
            row = analyze_slot(
                question_id=run_dir.name.split("-")[4],
                slot=slot,
                valid_ids=valid_ids,
                answerable=raw.get("answerable"),
                refusal_present=refusal_present,
                refusal_value=raw.get("refusal_reason"),
                finish_reason=envelope.get("finish_reason"),
                output_tokens=int((envelope.get("usage") or {}).get("output_tokens", 0)),
            )
            rows.append(row)
            question_slots.append(row)
        question_id = run_dir.name.split("-")[4]
        question_rows.append(
            {
                "question_id": question_id,
                "answerable": raw.get("answerable"),
                "slot_count": len(question_slots),
                "required_claim_ids_complete": {
                    row["required_claim_id"] for row in question_slots
                }
                == valid_ids,
                "all_other_fields_structurally_valid": all(
                    row["other_slot_fields_structurally_valid"]
                    for row in question_slots
                ),
                "all_content_shapes_v3_valid_without_status": all(
                    row["content_shape_v3_valid_without_status"]
                    for row in question_slots
                ),
                "refusal_reason_present": refusal_present,
                "extra_top_level_fields": sorted(
                    set(raw)
                    - {
                        "answerable",
                        "required_claim_results",
                        "refusal_reason",
                    }
                ),
            }
        )
    supported_with_empty_claim = sum(
        row["raw_status_value"] == "supported"
        and not (
            isinstance(row["claim_text_length"], int)
            and row["claim_text_length"] > 0
        )
        for row in rows
    )
    supported_with_nonempty_omission = sum(
        row["raw_status_value"] == "supported"
        and isinstance(row["omission_reason_value"], str)
        and bool(row["omission_reason_value"].strip())
        for row in rows
    )
    body = {
        "schema_version": "dev-v3-4-status-field-forensics-v1",
        "evaluation_version": "evidence-qa-dev-v3.4",
        "slot_count": len(rows),
        "status_distribution": dict(sorted(statuses.items())),
        "supported_count": statuses["supported"],
        "answerable_as_slot_status_count": statuses["answerable"],
        "answered_count": statuses["answered"],
        "unsupported_count": statuses["unsupported"],
        "missing_status_count": statuses["<missing>"],
        "other_unknown_status_count": sum(
            count
            for value, count in statuses.items()
            if value
            not in {
                "supported",
                "answerable",
                "answered",
                "unsupported",
                "not_applicable",
                "<missing>",
            }
        ),
        "status_excluded_structurally_valid_slots": sum(
            row["other_slot_fields_structurally_valid"] for row in rows
        ),
        "content_shape_v3_valid_slots": sum(
            row["content_shape_v3_valid_without_status"] for row in rows
        ),
        "supported_with_empty_claim_text": supported_with_empty_claim,
        "supported_with_nonempty_omission_reason": supported_with_nonempty_omission,
        "q001": next(row for row in question_rows if row["question_id"] == "q001"),
        "q013": next(row for row in question_rows if row["question_id"] == "q013"),
        "questions": question_rows,
        "slots": rows,
        "finding": (
            "All 27 slots retain the expected field names, types, and required-claim IDs "
            "apart from the semantic status enum. However, none already satisfies the "
            "new unique content-shape contract: most answered-looking slots use an exact "
            "empty omission_reason rather than null, while q015 contains claim/reason "
            "conflicts. Status removal alone therefore does not make the nine answerable "
            "historical payloads valid."
        ),
    }
    return body


def main() -> None:
    body = build()
    OUTPUT.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_DOC.write_text(
        "# Dev v3.4 Status Field Forensics\n\n"
        f"- Slots: {body['slot_count']}\n"
        f"- Raw statuses: `{body['status_distribution']}`\n"
        f"- Status-excluded structural fields valid: "
        f"{body['status_excluded_structurally_valid_slots']}/27\n"
        f"- New unique content shapes already valid: "
        f"{body['content_shape_v3_valid_slots']}/27\n"
        f"- Supported with empty claim text: "
        f"{body['supported_with_empty_claim_text']}\n"
        f"- Supported with non-empty omission reason: "
        f"{body['supported_with_nonempty_omission_reason']}\n\n"
        f"{body['finding']}\n",
        encoding="utf-8",
    )
    print(json.dumps(body, ensure_ascii=False))


if __name__ == "__main__":
    main()
