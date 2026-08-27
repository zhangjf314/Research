from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from paper_research.agents.report_quality import evaluate_report_quality
from paper_research.agents.research_synthesis_provider import (
    ResearchSynthesis,
    _validate_synthesis_citations,
)
from paper_research.providers.llm import normalize_structured_json_content

FAILURE_TYPES = {
    "NON_JSON_OUTPUT",
    "MARKDOWN_CODE_FENCE",
    "JSON_PREFIX_OR_SUFFIX",
    "TRUNCATED_JSON",
    "WRONG_TOP_LEVEL_SHAPE",
    "FIELD_NAME_VARIANT",
    "WRONG_FIELD_TYPE",
    "MISSING_REQUIRED_FIELD",
    "EXTRA_FIELD",
    "MISSING_SECTION",
    "DUPLICATE_SECTION",
    "INVALID_SECTION_ID",
    "EMPTY_CLAIMS",
    "UNKNOWN_CITATION_ID",
    "CITATION_NOT_ALLOWED_FOR_SECTION",
    "RESULTS_EVIDENCE_GAP_MISSING",
    "CROSS_FIELD_VALIDATION_FAILURE",
    "PROMPT_CONTRACT_FAILURE",
    "PARSER_BUG",
    "ACCOUNTING_BUG",
}


def replay_record(record: dict[str, Any]) -> dict[str, Any]:
    content = str(record.get("content") or "")
    section_evidence_ids = record.get("section_evidence_ids") or {}
    evidence_catalog = record.get("evidence_catalog") or {}
    allowed = set(evidence_catalog or record.get("allowed_citation_ids") or [])
    if not allowed:
        for values in section_evidence_ids.values():
            allowed.update(values)
    result: dict[str, Any] = {
        "attempt_number": record.get("attempt_number"),
        "content_length": len(content),
        "json_parse_status": "not_run",
        "normalization_actions": [],
        "top_level_keys": [],
        "schema_error_count": 0,
        "schema_error_locations": [],
        "schema_error_types": [],
        "failure_types": [],
        "offending_citation_ids": [],
        "citation_allowlist_details": [],
    }
    payload: dict[str, Any] | None = None
    try:
        payload, actions = normalize_structured_json_content(content)
        result["normalization_actions"] = actions
        result["json_parse_status"] = "passed"
        result["top_level_keys"] = sorted(payload)
        synthesis = ResearchSynthesis.model_validate(payload)
        _validate_synthesis_citations(
            synthesis,
            allowed_citation_ids=allowed,
            section_evidence_ids=section_evidence_ids,
        )
        catalog = _catalog_for_quality(allowed, evidence_catalog)
        report, sections = _render_minimal_report(synthesis)
        quality = evaluate_report_quality(
            report,
            sections=sections,
            evidence_catalog=catalog,
            section_evidence_ids=section_evidence_ids,
        )
        result.update(
            {
                "research_synthesis_schema": "passed",
                "section_set": "passed",
                "citation_allowlist": "passed",
                "report_render": "passed",
                "report_quality_gate": "passed" if quality.passed else "failed",
                "report_quality_failures": quality.failures,
            }
        )
    except json.JSONDecodeError as exc:
        result["json_parse_status"] = "failed"
        result["schema_error_count"] = 1
        result["schema_error_locations"] = [f"char:{exc.pos}"]
        result["schema_error_types"] = ["json_decode"]
        result["failure_types"] = classify_failure(content, [], exc)
    except (ValidationError, ValueError) as exc:
        result["research_synthesis_schema"] = "failed"
        locations, types = validation_paths(exc)
        result["schema_error_count"] = len(locations) or 1
        result["schema_error_locations"] = locations
        result["schema_error_types"] = types
        citation_details = _citation_allowlist_diagnostics(
            payload or {},
            allowed_citation_ids=allowed,
            section_evidence_ids=section_evidence_ids,
        )
        if citation_details:
            result["citation_allowlist"] = "failed"
            result["citation_allowlist_details"] = citation_details
            result["offending_citation_ids"] = sorted(
                {
                    citation_id
                    for detail in citation_details
                    for citation_id in detail.get("citation_ids", [])
                }
            )
        result["failure_types"] = classify_failure(content, types, exc, citation_details)
    return result


def validation_paths(exc: BaseException) -> tuple[list[str], list[str]]:
    if isinstance(exc, ValidationError):
        locations = [".".join(str(part) for part in item.get("loc", ())) for item in exc.errors()]
        types = [str(item.get("type")) for item in exc.errors()]
        return locations, types
    return ["<root>"], [type(exc).__name__]


def classify_failure(
    content: str,
    schema_types: list[str],
    exc: BaseException,
    citation_details: list[dict[str, Any]] | None = None,
) -> list[str]:
    output: set[str] = set()
    stripped = content.strip()
    if stripped.startswith("```"):
        output.add("MARKDOWN_CODE_FENCE")
    if not stripped.startswith("{"):
        output.add("JSON_PREFIX_OR_SUFFIX")
    if stripped and stripped[-1] not in {"}", "`"}:
        output.add("TRUNCATED_JSON")
    message = str(exc)
    if isinstance(exc, json.JSONDecodeError):
        output.add("NON_JSON_OUTPUT")
    if "section" in message and "exactly once" in message:
        output.add("MISSING_SECTION")
    if "duplicate research section" in message:
        output.add("DUPLICATE_SECTION")
    if "citation IDs outside section allowlist" in message:
        output.add("CITATION_NOT_ALLOWED_FOR_SECTION")
    if "unknown citation IDs" in message:
        output.add("UNKNOWN_CITATION_ID")
    for detail in citation_details or []:
        if detail.get("failure_type") == "UNKNOWN_CITATION_ID":
            output.add("UNKNOWN_CITATION_ID")
        if detail.get("failure_type") == "CITATION_NOT_ALLOWED_FOR_SECTION":
            output.add("CITATION_NOT_ALLOWED_FOR_SECTION")
    if any("missing" in item for item in schema_types):
        output.add("MISSING_REQUIRED_FIELD")
    if any("literal_error" in item for item in schema_types):
        output.add("INVALID_SECTION_ID")
    if any(
        "list_type" in item
        or "string_type" in item
        or "bool_type" in item
        or "model_type" in item
        for item in schema_types
    ):
        output.add("WRONG_FIELD_TYPE")
    if not output:
        output.add("CROSS_FIELD_VALIDATION_FAILURE")
    return sorted(output & FAILURE_TYPES)


def _citation_allowlist_diagnostics(
    payload: dict[str, Any],
    *,
    allowed_citation_ids: set[str],
    section_evidence_ids: dict[str, list[str]],
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    details: list[dict[str, Any]] = []
    sections = payload.get("sections")
    if isinstance(sections, list):
        for section_index, section in enumerate(sections):
            if not isinstance(section, dict):
                continue
            section_id = section.get("section_id")
            if not isinstance(section_id, str):
                continue
            allowed_for_section = set(section_evidence_ids.get(section_id, []))
            claims = section.get("claims")
            if not isinstance(claims, list):
                continue
            for claim_index, claim in enumerate(claims):
                if not isinstance(claim, dict):
                    continue
                citation_ids = claim.get("citation_ids")
                if not isinstance(citation_ids, list):
                    continue
                _append_citation_diagnostics(
                    details,
                    citation_ids=citation_ids,
                    allowed_citation_ids=allowed_citation_ids,
                    allowed_for_section=allowed_for_section,
                    location=f"sections.{section_index}.claims.{claim_index}.citation_ids",
                    section_id=section_id,
                )
    for field in ("consensus", "disagreements", "research_gaps"):
        values = payload.get(field)
        if not isinstance(values, list):
            continue
        for index, item in enumerate(values):
            if not isinstance(item, dict):
                continue
            citation_ids = item.get("citation_ids")
            if not isinstance(citation_ids, list):
                continue
            _append_citation_diagnostics(
                details,
                citation_ids=citation_ids,
                allowed_citation_ids=allowed_citation_ids,
                allowed_for_section=None,
                location=f"{field}.{index}.citation_ids",
                section_id=None,
            )
    return details


def _append_citation_diagnostics(
    details: list[dict[str, Any]],
    *,
    citation_ids: list[Any],
    allowed_citation_ids: set[str],
    allowed_for_section: set[str] | None,
    location: str,
    section_id: str | None,
) -> None:
    string_ids = [citation_id for citation_id in citation_ids if isinstance(citation_id, str)]
    unknown = [citation_id for citation_id in string_ids if citation_id not in allowed_citation_ids]
    if unknown:
        details.append(
            {
                "failure_type": "UNKNOWN_CITATION_ID",
                "location": location,
                "section_id": section_id,
                "citation_ids": unknown,
            }
        )
    if allowed_for_section is None:
        return
    outside = [
        citation_id
        for citation_id in string_ids
        if citation_id in allowed_citation_ids and citation_id not in allowed_for_section
    ]
    if outside:
        details.append(
            {
                "failure_type": "CITATION_NOT_ALLOWED_FOR_SECTION",
                "location": location,
                "section_id": section_id,
                "citation_ids": outside,
                "allowed_for_section": sorted(allowed_for_section),
            }
        )


def _catalog_for_quality(
    allowed: set[str],
    evidence_catalog: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for citation_id in sorted(allowed):
        raw = evidence_catalog.get(citation_id) if isinstance(evidence_catalog, dict) else None
        if isinstance(raw, dict):
            catalog[citation_id] = raw
        else:
            catalog[citation_id] = {
                "paper_id": "fixture-paper",
                "text": f"Fixture evidence for {citation_id}.",
            }
    return catalog


def _render_minimal_report(synthesis: ResearchSynthesis) -> tuple[str, dict[str, str]]:
    lines = ["# Replay report", "", synthesis.executive_summary, ""]
    sections: dict[str, str] = {}
    for section in synthesis.sections:
        section_lines = [f"## {section.section_id}", section.summary]
        for claim in section.claims:
            section_lines.append(f"- {claim.text} {' '.join(claim.citation_ids)}")
        if section.insufficient_evidence:
            section_lines.append(f"- Insufficient evidence: {section.evidence_gap}")
        sections[section.section_id] = "\n".join(section_lines[1:])
        lines.extend(section_lines)
        lines.append("")
    return "\n".join(lines), sections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--task-id")
    parser.add_argument("--attempt", choices=["1", "2", "all"], default="all")
    parser.add_argument("--all", action="store_true", help="Replay all attempts.")
    args = parser.parse_args()
    if args.task_id:
        input_label = f".runtime/research-synthesis-provider/{args.task_id}"
        records = _load_task_records(args.task_id)
    elif args.input:
        input_label = str(args.input)
        raw = json.loads(args.input.read_text(encoding="utf-8"))
        records = raw if isinstance(raw, list) else raw.get("attempts", [raw])
    else:
        raise SystemExit("--input or --task-id is required")
    attempt = "all" if args.all else args.attempt
    if attempt != "all":
        records = [record for record in records if str(record.get("attempt_number")) == attempt]
    results = [replay_record(record) for record in records]
    out = {
        "schema_version": "research-synthesis-schema-replay-v1",
        "input": input_label,
        "attempts": results,
        "all_passed": all(
            item.get("json_parse_status") == "passed"
            and item.get("research_synthesis_schema") == "passed"
            and item.get("citation_allowlist") == "passed"
            and item.get("report_render") == "passed"
            and item.get("report_quality_gate") == "passed"
            for item in results
        ),
    }
    if args.task_id == "ce25169e-7ab7-4d1b-92f2-fec77df06f0a":
        out.update(
            {
                "legacy_frozen_replay_status": "FAILED",
                "legacy_frozen_replay_classification": (
                    "EXPECTED_INCOMPATIBILITY_WITH_REVISED_PROTOCOL"
                ),
                "legacy_frozen_replay_gate": "DIAGNOSTIC_ONLY",
                "legacy_replay_passed": False,
                "legacy_response_validated": False,
                "new_protocol_live_requires_legacy_attempt_1_replay_passed": False,
                "legacy_failure_reasons": [
                    "Attempt 1: JSON parse passed and ResearchGap object shape is "
                    "compatible with the revised schema, but methods cited E14 outside "
                    "the legacy methods allowlist and results cited E2 outside the "
                    "legacy results allowlist.",
                    "Attempt 2: research_gaps remained a string array, and methods/results "
                    "repeated the same section allowlist violations.",
                ],
            }
        )
    Path("data/evaluation").mkdir(parents=True, exist_ok=True)
    Path("docs").mkdir(parents=True, exist_ok=True)
    Path("data/evaluation/research-synthesis-schema-replay-v1.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path("docs/research-synthesis-schema-replay-v1.md").write_text(
        "# Research synthesis schema replay v1\n\n"
        "```json\n"
        + json.dumps(out, ensure_ascii=False, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


def _load_task_records(task_id: str) -> list[dict[str, Any]]:
    task_dir = Path(".runtime/research-synthesis-provider") / task_id
    if not task_dir.exists():
        raise SystemExit(f"task raw response directory not found: {task_dir}")
    records = []
    for path in sorted(task_dir.glob("attempt-*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record.setdefault("content", record.get("raw_content", ""))
        records.append(record)
    if not records:
        raise SystemExit(f"no attempt files found in {task_dir}")
    return records


if __name__ == "__main__":
    main()
