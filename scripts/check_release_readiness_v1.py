"""Evaluate the repository's auditable RC or v1.0 release gates."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

STATUS_VALUES = {
    "passed",
    "partially_passed",
    "failed",
    "blocked",
    "not_run",
    "not_applicable",
}
STAGE_KEYS = {
    "stage_id",
    "title",
    "engineering_status",
    "evaluation_status",
    "production_gate",
    "evidence_files",
    "key_metrics",
    "blockers",
    "external_dependencies",
    "accepted_negative_result",
    "date",
    "git_commit_or_tag",
}
GATE_KEYS = {
    "gate_id",
    "description",
    "threshold",
    "measurement_source",
    "current_value",
    "status",
    "blocker_type",
    "remediation",
    "required_for_rc",
    "required_for_v1",
}
EVIDENCE_KEYS = {
    "artifact_path",
    "stage",
    "purpose",
    "immutable_or_generated",
    "authoritative_fields",
    "known_limitations",
    "supersedes",
    "superseded_by",
}
ISSUE_KEYS = {
    "issue_id",
    "severity",
    "component",
    "description",
    "impact",
    "workaround",
    "fix_plan",
    "owner",
    "target_stage",
    "release_blocker",
    "external_dependency",
    "evidence",
}
TASK_KEYS = {
    "task_id",
    "priority",
    "problem",
    "hypothesis",
    "implementation",
    "evaluation",
    "acceptance_threshold",
    "dependencies",
    "estimated_requests",
    "estimated_tokens",
    "human_review_required",
    "risk",
    "rollback",
    "target_release",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{line_number}")
        records.append(value)
    return records


def _validate_records(
    records: Any,
    required: set[str],
    label: str,
    errors: list[str],
) -> None:
    if not isinstance(records, list) or not records:
        errors.append(f"{label}: expected a non-empty list")
        return
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"{label}[{index}]: expected object")
            continue
        missing = sorted(required - record.keys())
        if missing:
            errors.append(f"{label}[{index}]: missing {missing}")


def _set_gate(gates: dict[str, dict[str, Any]], gate_id: str, passed: bool) -> None:
    if gate_id not in gates:
        raise ValueError(f"missing required gate definition: {gate_id}")
    if passed:
        gates[gate_id]["status"] = "passed"
    elif gates[gate_id]["status"] == "passed":
        gates[gate_id]["status"] = "failed"


def _apply_authoritative_checks(root: Path, gates: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    try:
        corpus = _load_json(root / "data/evaluation/production-corpus-v1.json")
        papers = corpus.get("papers", [])
        included = [paper for paper in papers if paper.get("included_in_production")]
        excluded_fixtures = [
            paper
            for paper in papers
            if not paper.get("included_in_production")
            and paper.get("corpus_role") == "ocr_fixture"
        ]
        _set_gate(gates, "DATA-01", corpus.get("manifest_version") == "production-corpus-v1")
        _set_gate(
            gates,
            "DATA-02",
            corpus.get("included_documents") == 34
            and len(included) == 34
            and corpus.get("excluded_ocr_fixtures") == 2,
        )
        _set_gate(gates, "DATA-03", len(excluded_fixtures) == 2)

        reranker = _load_json(root / "data/evaluation/reranker-ablation-v1.json")
        collection = reranker.get("collection", {})
        _set_gate(
            gates,
            "DATA-04",
            collection.get("paper_count") == 34
            and collection.get("point_count") == 2062
            and collection.get("dimension") == 1024,
        )

        gold = _load_jsonl(root / "data/evaluation/gold-set-v1.jsonl")
        _set_gate(
            gates,
            "DATA-05",
            len(gold) == 50 and all(row.get("review_status") == "approved" for row in gold),
        )
        retrieval_gold = _load_jsonl(root / "data/evaluation/retrieval-gold-v2.jsonl")
        signed_revision_states = {"approved", "not_required_scope_only"}
        _set_gate(
            gates,
            "DATA-06",
            len(retrieval_gold) == 50
            and all(row.get("review_status") == "approved" for row in retrieval_gold)
            and all(
                row.get("query_revision_review_status") in signed_revision_states
                for row in retrieval_gold
            ),
        )

        config_text = (root / "src/paper_research/config.py").read_text(encoding="utf-8")
        reranker_disabled = bool(
            re.search(r"rerank_enabled\s*:\s*bool\s*=\s*False", config_text)
        )
        _set_gate(gates, "RERANK-01", reranker_disabled)

        qa = _load_json(root / "data/evaluation/qa-production-v1.json")
        metrics = qa.get("metrics", {})
        _set_gate(
            gates,
            "QA-01",
            metrics.get("json_parse_success_rate", 0) >= 0.99
            and metrics.get("schema_validation_success_rate", 0) >= 0.99,
        )
        _set_gate(gates, "QA-02", metrics.get("answerable_accuracy", 0) >= 0.90)
        _set_gate(gates, "QA-03", metrics.get("refusal_accuracy", 0) >= 0.95)
        _set_gate(gates, "QA-04", metrics.get("required_claim_coverage", 0) >= 0.80)
        _set_gate(gates, "QA-05", metrics.get("citation_precision", 0) >= 0.90)
        _set_gate(gates, "QA-06", metrics.get("citation_recall", 0) >= 0.80)
        latency = metrics.get("total_latency", {})
        _set_gate(
            gates,
            "QA-09",
            metrics.get("total_tokens", 0) > 0
            and latency.get("mean_ms") is not None
            and latency.get("p95_ms") is not None,
        )

        citation_audit = _load_json(
            root / "data/evaluation/citation-human-audit-summary-v1.json"
        )
        support = citation_audit.get("overall", {})
        _set_gate(
            gates,
            "QA-08",
            support.get("strict_support_rate", 0) >= 0.80
            and support.get("lenient_support_rate", 0) >= 0.90,
        )

        deep_audit = _load_json(root / "data/evaluation/stage11d-final-audit-v1.json")
        runs = deep_audit.get("runs", [])
        selected = [row for row in runs if row.get("selected_by_latest_successful")]
        _set_gate(
            gates,
            "DR-01",
            len(selected) == 3
            and {row.get("question_id") for row in selected} == {"q003", "q005", "q049"},
        )
        _set_gate(
            gates,
            "DR-05",
            len(runs) == 7 and all(row.get("run_directory_complete") for row in runs),
        )
        failed = [row for row in runs if row.get("status") == "provider_failed"]
        _set_gate(
            gates,
            "DR-06",
            len(failed) == 3
            and all(row.get("active_reservation", 0) > 0 for row in failed),
        )
        checks = deep_audit.get("checks", {})
        _set_gate(
            gates,
            "OPS-03",
            checks.get("api_key_value_hits") == 0
            and checks.get("authorization_header_hits") == 0,
        )

        test_report = _load_json(root / "data/evaluation/stage12-test-report-v1.json")
        commands = test_report.get("commands", [])
        _set_gate(
            gates,
            "OPS-11",
            test_report.get("all_passed") is True
            and test_report.get("pytest", {}).get("failed") == 0
            and len(commands) == 4
            and all(command.get("exit_code") == 0 for command in commands),
        )

        pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
        version_match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
        package_version = version_match.group(1) if version_match else None
        _set_gate(gates, "REL-01", package_version == "0.9.0rc3")
        _set_gate(gates, "REL-02", package_version == "1.0.0")

        summary = _load_json(root / "data/evaluation/deep-research-smoke-v1.json")
        failed_ids = {row.get("run_id") for row in summary.get("failed_attempts", [])}
        expected_failed = {row.get("run_id") for row in failed}
        if not expected_failed.issubset(failed_ids):
            errors.append("latest-successful summary hides one or more failed attempts")

        diagnostics = _load_json(root / "data/evaluation/qa-context-diagnostics-v1.json")
        oracle_rows = [row for row in diagnostics.get("runs", []) if row.get("oracle")]
        if not oracle_rows or any(row.get("production_metric") for row in oracle_rows):
            errors.append("Oracle diagnostics are missing or counted as Production metrics")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"authoritative artifact validation failed: {exc}")
    return errors


def evaluate(root: Path, target: str, strict: bool) -> tuple[dict[str, Any], int]:
    root = root.resolve()
    paths = {
        "stages": root / "data/evaluation/project-stage-status-v1.json",
        "gates": root / "data/evaluation/v1-release-gates.json",
        "evidence": root / "data/evaluation/evaluation-evidence-index.json",
        "issues": root / "data/evaluation/known-issues-v1.json",
        "plan": root / "data/evaluation/v1-gap-closure-plan.json",
    }
    errors: list[str] = []
    documents: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        if not path.is_file():
            errors.append(f"missing artifact: {path.relative_to(root)}")
            continue
        try:
            documents[label] = _load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid artifact {path.relative_to(root)}: {exc}")

    if len(documents) == len(paths):
        _validate_records(documents["stages"].get("stages"), STAGE_KEYS, "stages", errors)
        _validate_records(documents["gates"].get("gates"), GATE_KEYS, "gates", errors)
        _validate_records(
            documents["evidence"].get("artifacts"), EVIDENCE_KEYS, "evidence", errors
        )
        _validate_records(documents["issues"].get("issues"), ISSUE_KEYS, "issues", errors)
        _validate_records(documents["plan"].get("tasks"), TASK_KEYS, "tasks", errors)

    gate_rows = documents.get("gates", {}).get("gates", [])
    gate_map = {row.get("gate_id"): dict(row) for row in gate_rows if isinstance(row, dict)}
    if len(gate_map) != len(gate_rows):
        errors.append("gate IDs must be present and unique")
    if gate_map:
        errors.extend(_apply_authoritative_checks(root, gate_map))

    for stage in documents.get("stages", {}).get("stages", []):
        for field in ("engineering_status", "evaluation_status", "production_gate"):
            if stage.get(field) not in STATUS_VALUES:
                errors.append(f"{stage.get('stage_id')}: invalid {field}")
        for evidence in stage.get("evidence_files", []):
            if not (root / evidence).exists():
                errors.append(f"missing stage evidence: {evidence}")

    for gate in gate_map.values():
        if gate.get("status") not in STATUS_VALUES:
            errors.append(f"{gate.get('gate_id')}: invalid status")
        source = gate.get("measurement_source")
        if isinstance(source, str) and "/" in source and not (root / source).exists():
            errors.append(f"missing gate evidence: {source}")

    required_field = "required_for_rc" if target == "rc" else "required_for_v1"
    required = [gate for gate in gate_map.values() if gate.get(required_field) is True]
    unmet = [gate for gate in required if gate.get("status") != "passed"]
    rc_required = [gate for gate in gate_map.values() if gate.get("required_for_rc") is True]
    rc_unmet = [gate for gate in rc_required if gate.get("status") != "passed"]
    if strict and errors:
        result_status = "failed"
    elif unmet:
        result_status = "failed"
    else:
        result_status = "passed"

    recommended_rc = documents.get("gates", {}).get("recommended_rc_version")
    output = {
        "schema_version": "release-readiness-result-v1",
        "target": target,
        "strict": strict,
        "status": result_status,
        "recommended_rc_version": recommended_rc,
        "highest_satisfied_version": (
            recommended_rc if not rc_unmet and not errors else "v0.9.0-rc2"
        ),
        "production_ready": target == "v1" and result_status == "passed",
        "required_gate_count": len(required),
        "passed_gate_count": len(required) - len(unmet),
        "unmet_gates": [
            {
                "gate_id": gate["gate_id"],
                "status": gate["status"],
                "blocker_type": gate["blocker_type"],
                "current_value": gate["current_value"],
                "remediation": gate["remediation"],
            }
            for gate in unmet
        ],
        "validation_errors": errors,
        "gates": [gate_map[key] for key in sorted(gate_map)],
    }
    return output, 0 if result_status == "passed" else 1


def _render_text(result: dict[str, Any]) -> str:
    lines = [
        f"Target: {result['target']}",
        f"Status: {result['status'].upper()}",
        f"Recommended RC: {result['recommended_rc_version']}",
        f"Highest satisfied version: {result['highest_satisfied_version']}",
        f"Required gates: {result['passed_gate_count']}/{result['required_gate_count']} passed",
        f"Production-ready: {'YES' if result['production_ready'] else 'NO'}",
    ]
    if result["unmet_gates"]:
        lines.append("Unmet gates:")
        lines.extend(
            f"- {gate['gate_id']}: {gate['status']} ({gate['blocker_type']})"
            for gate in result["unmet_gates"]
        )
    if result["validation_errors"]:
        lines.append("Validation errors:")
        lines.extend(f"- {error}" for error in result["validation_errors"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("rc", "v1"), required=True)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    result, exit_code = evaluate(root, args.target, args.strict)
    rendered = (
        json.dumps(result, ensure_ascii=False, indent=2)
        if args.format == "json"
        else _render_text(result)
    )
    if args.output:
        output = args.output if args.output.is_absolute() else root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
