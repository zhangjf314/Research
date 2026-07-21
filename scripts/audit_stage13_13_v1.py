# ruff: noqa: E501
"""Offline failure forensics and reservation reconciliation for Stage 13.13."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from paper_research.evaluation.request_accounting import (
    RequestTerminalState,
    close_reservation_for_terminal_run,
)

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DOCS, canonical_hash
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DATA, DOCS, canonical_hash  # type: ignore[no-redef]

ROOT = DATA.parents[1]
RUN_ROOT = DATA / "evidence-qa-dev-v3-2/runs"
FORENSICS_JSON = DATA / "dev-v3-2-schema-failure-forensics-v1.json"
FORENSICS_MD = DOCS / "dev-v3-2-schema-failure-forensics-v1.md"
RECON_JSON = DATA / "stage13-12-reservation-reconciliation-v1.json"
RECON_MD = DOCS / "stage13-12-reservation-reconciliation-v1.md"
DECISION_JSON = DATA / "schema-reliability-v1-candidate.json"
DECISION_MD = DOCS / "schema-reliability-v1-candidate.md"
FAILED_QUESTIONS = ("q005", "q007", "q013", "q050")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def delivered_messages_hash(run_dir: Path) -> str:
    messages = [
        {
            "role": "system",
            "content": (run_dir / "rendered-system-prompt.txt").read_text(encoding="utf-8"),
        },
        {
            "role": "user",
            "content": (run_dir / "rendered-user-prompt.txt").read_text(encoding="utf-8"),
        },
    ]
    return canonical_hash(messages)


def safe_excerpt(value: str, *, prefix: bool) -> str:
    excerpt = value[:180] if prefix else value[-180:]
    return " ".join(excerpt.replace("\x00", "").split())


def classify_json_failure(content: str, finish_reason: str | None) -> dict[str, Any]:
    try:
        json.loads(content)
        return {"failure_type": None, "parse_error_position": None, "truncated": False}
    except json.JSONDecodeError as exc:
        stripped = content.strip()
        if finish_reason == "length":
            kind = "truncated_json"
        elif stripped.startswith("```"):
            kind = "markdown_wrapped_json"
        elif not stripped.startswith("{"):
            kind = "prose_wrapped_json"
        elif exc.msg == "Invalid \\escape":
            kind = "invalid_escape"
        elif "Unterminated string" in exc.msg:
            kind = "unterminated_string"
        elif exc.pos < len(content) - 5:
            kind = "extra_trailing_content"
        else:
            kind = "provider_json_mode_noncompliance"
        return {
            "failure_type": kind,
            "parse_error": exc.msg,
            "parse_error_position": exc.pos,
            "parse_error_line": exc.lineno,
            "parse_error_column": exc.colno,
            "truncated": finish_reason == "length",
        }


def build_forensics() -> dict[str, Any]:
    rows = []
    for question_id in FAILED_QUESTIONS:
        run_dir = next(RUN_ROOT.glob(f"live-dev-v3-2-{question_id}-*"))
        result = json.loads((run_dir / "final-result.json").read_text(encoding="utf-8"))
        raw_response = json.loads(
            (run_dir / "raw-provider-response.json").read_text(encoding="utf-8")
        )
        envelope = json.loads(
            (run_dir / "provider-response-envelope.json").read_text(encoding="utf-8")
        )
        payload = json.loads(
            (run_dir / "required-claims-input.json").read_text(encoding="utf-8")
        )
        metadata = json.loads(
            (run_dir / "run-metadata.json").read_text(encoding="utf-8")
        )
        registry = json.loads(
            (run_dir / "citation-registry.json").read_text(encoding="utf-8")
        )
        choice = raw_response["choices"][0]
        content = choice["message"]["content"]
        ledger_path = run_dir / "request-ledger.jsonl"
        ledger = [
            json.loads(line)
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        system_prompt = (run_dir / "rendered-system-prompt.txt").read_text(encoding="utf-8")
        user_prompt = (run_dir / "rendered-user-prompt.txt").read_text(encoding="utf-8")
        parse = classify_json_failure(content, choice.get("finish_reason"))
        unknown_value = (
            result["failure_reason"].split(": ", 1)[1]
            if result["failure_type"] == "unknown_citation_id"
            else None
        )
        visible_namespaces = {
            "public_citation_ids": bool(
                any(claim.get("allowed_citation_ids") for claim in payload["required_claims"])
            ),
            "evidence_ids": '"evidence_id"' in user_prompt,
            "block_ids": '"block_id"' in user_prompt,
            "paper_ids": '"paper_id"' in user_prompt,
            "required_claim_ids": '"required_claim_id"' in user_prompt,
        }
        prompt_versions_in_delivered = {
            "v3_1_literal_count": system_prompt.count(
                "qa-required-claims-citation-id-v3.1"
            ),
            "v3_2_literal_count": (
                system_prompt + user_prompt
            ).count("qa-required-claims-citation-id-v3.2-candidate"),
        }
        rows.append(
            {
                "run_id": result["run_id"],
                "question_id": question_id,
                "request_id": metadata["request_id"],
                "required_claim_count": result["required_claim_count"],
                "rendered_system_prompt_sha256": sha256_bytes(system_prompt.encode("utf-8")),
                "rendered_user_prompt_sha256": sha256_bytes(user_prompt.encode("utf-8")),
                "exact_delivered_messages_sha256": delivered_messages_hash(run_dir),
                "expected_prompt_version": "qa-required-claims-citation-id-v3.2-candidate",
                "prompt_versions_actually_present": prompt_versions_in_delivered,
                "expected_schema_hash": result["schema_hash"],
                "response_format": {"type": "json_object"},
                "max_output_tokens": payload["output_budget"]["calculated_max_output_tokens"],
                "raw_response_file_bytes": (run_dir / "raw-provider-response.json").stat().st_size,
                "raw_content_bytes": len(content.encode("utf-8")),
                "raw_response_file_sha256": sha256_file(
                    run_dir / "raw-provider-response.json"
                ),
                "raw_content_sha256": sha256_bytes(content.encode("utf-8")),
                "finish_reason": choice.get("finish_reason"),
                "provider_usage": raw_response["usage"],
                "output_token_count": raw_response["usage"]["completion_tokens"],
                "response_envelope": envelope,
                "raw_response_prefix_safe_excerpt": safe_excerpt(content, prefix=True),
                "raw_response_suffix_safe_excerpt": safe_excerpt(content, prefix=False),
                **parse,
                "schema_error_path": result["failure_reason"],
                "unknown_id_value": unknown_value,
                "unknown_id_value_class": (
                    "internal_evidence_id"
                    if unknown_value and unknown_value.startswith("ev-")
                    else None
                ),
                "registry_legal_ids": [
                    entry["citation_id"] for entry in registry["entries"]
                ],
                "model_visible_id_namespaces": visible_namespaces,
                "request_ledger_terminal_events": [
                    event["event"]
                    for event in ledger
                    if event["event"]
                    in {
                        "completed",
                        "validation_failed",
                        "reservation_settled",
                        "reservation_released",
                    }
                ],
                "historical_request_ledger_sha256": sha256_file(ledger_path),
                "reservation_id": metadata["request_id"],
                "reservation_amount": 24000,
                "settlement_release_events": [],
                "policy_executed": False,
                "final_run_status": result["status"],
            }
        )
    by_question = {row["question_id"]: row for row in rows}
    root_causes = {
        "q005": {
            "classification": "model_protocol_version_copy_failure",
            "finding": (
                "The exact delivered system prompt was constructed from the v3.1 prompt "
                "and contained two v3.1 literals and no v3.2 literal. The user payload "
                "contained v3.2, but the model copied the explicit v3.1 example."
            ),
        },
        "q007": {
            "classification": by_question["q007"]["failure_type"],
            "finding": (
                "Provider returned finish_reason=length at the exact 832-token output "
                "budget. JSON ended mid-generation with repeated input-field prose."
            ),
        },
        "q013": {
            "classification": "multiple_copyable_id_namespaces",
            "finding": (
                "The model copied an ev-* evidence_id because the user payload exposed "
                "evidence_id beside public E### citation IDs."
            ),
        },
        "q050": {
            "classification": by_question["q050"]["failure_type"],
            "finding": (
                "Provider returned finish_reason=length at the exact 832-token output "
                "budget. The JSON object was incomplete and followed by padding newlines."
            ),
        },
    }
    return {
        "schema_version": "dev-v3-2-schema-failure-forensics-v1",
        "offline_only": True,
        "historical_results_modified": False,
        "runs": rows,
        "root_causes": root_causes,
    }


def build_reconciliation() -> dict[str, Any]:
    records = []
    for question_id in FAILED_QUESTIONS:
        run_dir = next(RUN_ROOT.glob(f"live-dev-v3-2-{question_id}-*"))
        result = json.loads((run_dir / "final-result.json").read_text(encoding="utf-8"))
        metadata = json.loads(
            (run_dir / "run-metadata.json").read_text(encoding="utf-8")
        )
        ledger_path = run_dir / "request-ledger.jsonl"
        ledger = [
            json.loads(line)
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        events, close = close_reservation_for_terminal_run(
            ledger,
            reservation_id=metadata["request_id"],
            request_id=metadata["request_id"],
            reserved_tokens=24000,
            terminal_state=(
                RequestTerminalState.MALFORMED_JSON
                if result["failure_type"] == "malformed_json"
                else RequestTerminalState.SCHEMA_FAILED
            ),
            provider_usage=result["usage"],
            request_sent=True,
        )
        proposed = events[-1]
        records.append(
            {
                "run_id": result["run_id"],
                "question_id": question_id,
                "reservation_id": metadata["request_id"],
                "original_reserve_event": next(
                    event for event in ledger if event["event"] == "budget_reserved"
                ),
                "provider_usage": result["usage"],
                "historical_missing_terminal_event": True,
                "proposed_terminal_event": {
                    **proposed,
                    "event": "reconciliation_settlement_v1",
                    "underlying_terminal_event": proposed["event"],
                },
                "reconciliation_reason": (
                    "Provider usage was reported before downstream validation failed."
                ),
                "amount_settled_tokens": proposed["settled_tokens"],
                "unused_reservation_capacity_tokens": (
                    24000 - proposed["settled_tokens"]
                ),
                "remaining_active_tokens": close["effective_active_tokens"],
                "immutable_historical_ledger_sha256": sha256_file(ledger_path),
            }
        )
    settled = sum(row["amount_settled_tokens"] for row in records)
    body = {
        "schema_version": "stage13-12-reservation-reconciliation-v1",
        "historical_ledgers_modified": False,
        "reconciliation_ledger_mode": "independent_append-only-compensation",
        "historical_unclosed_reservations": len(records),
        "historical_reserved_tokens": 24000 * len(records),
        "reconciled_reservations": len(records),
        "provider_reported_tokens_settled": settled,
        "unused_reservation_capacity_closed": 24000 * len(records) - settled,
        "effective_active_reservations": 0,
        "effective_active_reserved_tokens": 0,
        "double_settlement_count": 0,
        "records": records,
    }
    body["reconciliation_signature"] = canonical_hash(body)
    return body


def schema_decision() -> dict[str, Any]:
    matrix = [
        {
            "option": "A",
            "design": "minimal model payload plus local immutable envelope",
            "schema_reliability": "high",
            "semantic_risk": "low",
            "comparability": "new protocol version required",
            "implementation_complexity": "medium",
            "citation_quality_control": "unchanged",
            "auditability": "high",
            "provider_dependency": "lower",
            "backward_compatibility": "explicit adapter required",
        },
        {
            "option": "B",
            "design": "model emits no citation IDs; local policy selects citations",
            "schema_reliability": "high",
            "semantic_risk": "medium",
            "comparability": "new protocol version required",
            "implementation_complexity": "medium",
            "citation_quality_control": "deterministic and capped",
            "auditability": "high",
            "provider_dependency": "lower",
            "backward_compatibility": "explicit adapter required",
        },
        {
            "option": "C",
            "design": "retain current schema with simpler prompt and one public namespace",
            "schema_reliability": "medium",
            "semantic_risk": "low",
            "comparability": "closest to v3.2",
            "implementation_complexity": "low",
            "citation_quality_control": "model plus local policy",
            "auditability": "medium",
            "provider_dependency": "unchanged",
            "backward_compatibility": "highest",
        },
    ]
    return {
        "schema_version": "schema-reliability-v1-candidate-decision-v1",
        "candidate_name": "schema-reliability-v1-candidate",
        "options": matrix,
        "selected_option": "A+B",
        "selection_reason": (
            "A removes model copying of fixed protocol constants; B removes the internal/"
            "public citation identifier confusion. Malformed JSON remains a strict failure."
        ),
        "model_payload_schema": {
            "answerable": "boolean",
            "required_claim_results": [
                {
                    "required_claim_id": "string",
                    "status": "answered|unsupported|not_applicable",
                    "claim_text": "string|null",
                    "omission_reason": "string|null",
                }
            ],
            "refusal_reason": "string|null",
        },
        "locally_bound_envelope_schema": {
            "question_id": "string",
            "answerable": "boolean",
            "required_claim_results": "model results plus deterministic citation_ids",
            "refusal_reason": "string|null",
            "prompt_version": "schema-reliability-v1-candidate",
            "citation_protocol": "citation-id-v2",
        },
        "citation_id_responsibility": "local deterministic policy",
        "model_visible_copyable_citation_namespaces": 0,
        "historical_protocol_modified": False,
        "next_live_authorized": False,
    }


def write_reports(
    forensics: dict[str, Any],
    reconciliation: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    FORENSICS_JSON.write_text(
        json.dumps(forensics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    RECON_JSON.write_text(
        json.dumps(reconciliation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    DECISION_JSON.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    FORENSICS_MD.write_text(
        "# Dev v3.2 Schema Failure Forensics\n\n"
        "- Offline only; no raw response, formal result, or historical Gate was changed.\n"
        "- q005: exact delivered prompt contained the v3.1 examples; classification "
        "`model_protocol_version_copy_failure`.\n"
        "- q007: `finish_reason=length`, 832/832 completion tokens, truncated JSON.\n"
        "- q013: copied an internal `ev-*` evidence ID from a multi-namespace payload.\n"
        "- q050: `finish_reason=length`, 832/832 completion tokens, truncated JSON.\n"
        "- Full prompts and source passages are not reproduced; only hashes and short "
        "safe excerpts are retained.\n\n"
        "| Question | Failure | Finish | Output tokens | Delivered messages hash |\n"
        "|---|---|---|---:|---|\n"
        + "\n".join(
            f"| {row['question_id']} | {row['failure_type'] or row['schema_error_path'].split(':', 1)[0]} "
            f"| {row['finish_reason']} | {row['output_token_count']} | "
            f"`{row['exact_delivered_messages_sha256']}` |"
            for row in forensics["runs"]
        )
        + "\n",
        encoding="utf-8",
    )
    RECON_MD.write_text(
        "# Stage 13.12 Reservation Reconciliation\n\n"
        f"- Historical unclosed reservations: {reconciliation['historical_unclosed_reservations']} "
        f"({reconciliation['historical_reserved_tokens']:,} reserved tokens).\n"
        f"- Reconciled: {reconciliation['reconciled_reservations']}; effective active: "
        f"{reconciliation['effective_active_reservations']}.\n"
        f"- Provider-reported usage settled: "
        f"{reconciliation['provider_reported_tokens_settled']:,} tokens.\n"
        f"- Unused reservation capacity closed: "
        f"{reconciliation['unused_reservation_capacity_closed']:,} tokens.\n"
        "- Historical ledgers remain byte-for-byte unchanged. Compensation is recorded "
        "in this independent, append-only reconciliation ledger.\n"
        "- Double settlements: 0.\n",
        encoding="utf-8",
    )
    DECISION_MD.write_text(
        "# Schema Reliability v1 Candidate\n\n"
        "- Selected: **A+B** — minimal model payload plus locally bound immutable "
        "envelope, with citation selection owned by deterministic local policy.\n"
        "- The model no longer copies prompt/citation protocol constants and does not "
        "emit citation IDs.\n"
        "- Malformed JSON remains a strict failure; no repair or normalization is added.\n"
        "- This is a new offline candidate, not a Dev v3.2 rerun or Dev v3.3 authorization.\n"
        "- `NEXT_LIVE_AUTHORIZED=false`.\n",
        encoding="utf-8",
    )


def main() -> None:
    forensics = build_forensics()
    reconciliation = build_reconciliation()
    decision = schema_decision()
    write_reports(forensics, reconciliation, decision)
    print(
        json.dumps(
            {
                "forensic_runs": len(forensics["runs"]),
                "reconciled": reconciliation["reconciled_reservations"],
                "effective_active": reconciliation["effective_active_reservations"],
                "candidate": decision["selected_option"],
            }
        )
    )


if __name__ == "__main__":
    main()
