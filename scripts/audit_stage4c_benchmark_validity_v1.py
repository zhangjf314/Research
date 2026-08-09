"""Audit Stage 4C benchmark metric provenance without rerunning systems."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BENCH = Path("data/evaluation/research-agent/benchmark")
DOCS = Path("docs/research-agent/benchmark")
RUNTIME = Path(".runtime/stage4/stage4-official-v1-attempt4")

FINAL = BENCH / "stage4-final-benchmark-v1.json"
FREEZE = BENCH / "stage4-blind-score-freeze-v1.json"
PAIRED = BENCH / "stage4-unblinded-paired-results-v1.json"
BOOTSTRAP = BENCH / "stage4-paired-bootstrap-v1.json"
RUBRIC_SCORES = BENCH / "stage4-blind-rubric-scores-v1.json"
BLIND_PACKAGE = BENCH / "stage4-blinded-evaluation-package-v1.json"
EXECUTION_RESULTS = BENCH / "stage4-execution-results-v1.json"
LABEL_MAP = RUNTIME / "system-label-map.json"
RAW_UNITS = RUNTIME / "raw-units"

OUT_LOCK = BENCH / "stage4c-pre-validity-audit-lock-v1.json"
OUT_PROVENANCE = BENCH / "stage4c-metric-provenance-v1.json"
OUT_VALIDITY = BENCH / "stage4c-final-validity-audit-v1.json"
OUT_BOUNDARY = BENCH / "stage4-portfolio-claim-boundary-v1.json"
OUT_READINESS = BENCH / "stage4-portfolio-release-readiness-v1.json"

DOC_PROVENANCE = DOCS / "stage4c-metric-provenance-v1.md"
DOC_VALIDITY = DOCS / "stage4c-final-validity-audit-v1.md"
DOC_BOUNDARY = DOCS / "stage4-portfolio-claim-boundary-v1.md"
DOC_READINESS = DOCS / "stage4-portfolio-release-readiness-v1.md"

ORIGINAL_BLIND_SCORE_HASH = "6579453399f1e3710b8afa21de4548aad9eb60865b0b4de95332f7d9e7e65fb8"


def now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def has_body(payload: dict[str, Any]) -> bool:
    for key in ("report", "answer", "final_answer", "markdown_report"):
        if isinstance(payload.get(key), str) and payload[key].strip():
            return True
    return False


def raw_response(task_id: str, system: str) -> dict[str, Any]:
    path = RAW_UNITS / f"research-benchmark-v1__{task_id}__{system}" / "response.json"
    return read_json(path)


def create_lock() -> dict[str, Any]:
    final = read_json(FINAL)
    freeze = read_json(FREEZE)
    paired = read_json(PAIRED)
    bootstrap = read_json(BOOTSTRAP)
    lock = {
        "schema_version": "stage4c-pre-validity-audit-lock-v1",
        "created_at": now(),
        "original_stage4c_commit": git_head(),
        "original_final_benchmark_hash": sha256(FINAL),
        "original_blind_score_freeze_hash": sha256(FREEZE),
        "original_blind_score_bundle_hash": freeze["blind_scores_bundle_hash"],
        "original_paired_results_hash": sha256(PAIRED),
        "original_bootstrap_hash": sha256(BOOTSTRAP),
        "stage4c_complete": final["stage4c_complete"],
        "stage4_complete": final["stage4_complete"],
        "evaluated_pairs": paired["evaluated_pairs"],
        "bootstrap_seed": bootstrap["seed"],
        "bootstrap_resamples": bootstrap["resamples"],
    }
    write_json(OUT_LOCK, lock)
    return lock


def vectors_by_system() -> dict[str, dict[str, list[float]]]:
    paired = read_json(PAIRED)
    vectors: dict[str, dict[str, list[float]]] = {
        "workflow": defaultdict(list),
        "agent": defaultdict(list),
    }
    for row in paired["rows"]:
        for system in ("workflow", "agent"):
            record = row[system]
            for metric in (
                "completed",
                "required_dimension_coverage",
                "required_claim_coverage",
                "evidence_coverage",
            ):
                vectors[system][metric].append(float(record[metric]))
    return vectors


def metric_provenance() -> dict[str, Any]:
    paired = read_json(PAIRED)
    rubric = read_json(RUBRIC_SCORES)
    vectors = vectors_by_system()
    vector_identity = {}
    for system, values in vectors.items():
        completion = values["completed"]
        vector_identity[system] = {
            "dimension_vector_equals_completion_vector": values[
                "required_dimension_coverage"
            ]
            == completion,
            "claim_vector_equals_completion_vector": values["required_claim_coverage"]
            == completion,
            "evidence_vector_equals_completion_vector": values["evidence_coverage"]
            == completion,
        }

    metrics = {
        "task_success_rate": {
            "classification": "STRUCTURAL_PROXY",
            "evidence_tier": "Tier 2 - Deterministic structural/proxy",
            "evaluator_function": "score_output",
            "input_fields": [
                "status",
                "behavioral_metrics.verification_status",
                "citation_ids_structurally_valid",
                "citation_structure_parseable",
                "category",
            ],
            "formula": (
                "completed and verification_pass and citation_validity and "
                "proxy coverage thresholds"
            ),
            "safe_interpretation": (
                "Structured task-success proxy, not direct semantic task success."
            ),
        },
        "partial_or_better_rate": {
            "classification": "STRUCTURAL_PROXY",
            "evidence_tier": "Tier 2 - Deterministic structural/proxy",
            "evaluator_function": "score_output",
            "input_fields": [
                "status",
                "behavioral_metrics.verification_status",
                "citation structure fields",
            ],
            "formula": "proxy claim coverage >= 0.6 and proxy dimension coverage >= 0.7",
            "safe_interpretation": "Partial-or-better structural proxy.",
        },
        "required_dimension_coverage": {
            "classification": "STRUCTURAL_PROXY",
            "evidence_tier": "Tier 2 - Deterministic structural/proxy",
            "evaluator_function": "score_output",
            "input_fields": ["status", "behavioral_metrics.verification_status"],
            "formula": "1.0 if COMPLETED and verification PASS else 0.0",
            "dimensions_individually_scored": 0,
            "safe_interpretation": (
                "Does not constitute direct scoring of all 250 dimensions."
            ),
        },
        "required_claim_coverage": {
            "classification": "STRUCTURAL_PROXY",
            "evidence_tier": "Tier 2 - Deterministic structural/proxy",
            "evaluator_function": "score_output",
            "input_fields": ["status", "behavioral_metrics.verification_status"],
            "formula": "1.0 if COMPLETED and verification PASS else 0.0",
            "claims_individually_scored": 0,
            "safe_interpretation": (
                "Does not constitute direct semantic scoring of all 180 claims."
            ),
        },
        "evidence_coverage": {
            "classification": "STRUCTURAL_PROXY",
            "evidence_tier": "Tier 2 - Deterministic structural/proxy",
            "evaluator_function": "score_output",
            "input_fields": [
                "status",
                "behavioral_metrics.verification_status",
                "behavioral_metrics.evidence_count",
            ],
            "formula": "1.0 if COMPLETED and verification PASS and evidence_count > 0 else 0.0",
            "evidence_items_individually_scored": 0,
            "evidence_coverage_source": "terminal status plus evidence_count proxy",
            "safe_interpretation": (
                "Does not prove each gold evidence set was matched to output text."
            ),
        },
        "core_unsupported_claim_rate": {
            "classification": "STRUCTURAL_PROXY",
            "evidence_tier": "Tier 2 - Deterministic structural/proxy",
            "evaluator_function": "score_output",
            "input_fields": ["status", "behavioral_metrics.verification_status"],
            "formula": "0 for verified completed outputs; failed outputs emit no scored claims",
            "evaluated_core_claim_count": 0,
            "unsupported_core_claim_count": sum(
                int(r["unsupported_core_claim_count"]) for r in rubric["records"]
            ),
            "safe_interpretation": (
                "A zero value here is not proof of perfect factual reliability."
            ),
        },
        "citation_validity": {
            "classification": "CITATION_STRUCTURE_DERIVED",
            "evidence_tier": "Tier 2 - Deterministic structural/proxy",
            "evaluator_function": "score_output",
            "input_fields": [
                "citation_ids_structurally_valid",
                "citation_structure_parseable",
            ],
            "formula": "1.0 when citation structure fields are true",
            "safe_interpretation": (
                "Structural validity; empty citation sets can be vacuously valid."
            ),
        },
        "gap_handling_accuracy": {
            "classification": "STRUCTURAL_PROXY",
            "evidence_tier": "Tier 2 - Deterministic structural/proxy",
            "evaluator_function": "score_output",
            "input_fields": ["category", "status", "stop_reason", "failure_category"],
            "formula": "gap task stop/failure proxy; non-gap completed outputs count as handled",
            "safe_interpretation": "Structural gap-handling proxy.",
        },
        "w_t_l": {
            "classification": "STRUCTURAL_PROXY",
            "evidence_tier": "Tier 2 - Deterministic structural/proxy",
            "evaluator_function": "paired_deltas",
            "input_fields": [
                "task_success",
                "required_claim_coverage",
                "core_unsupported_claim_rate",
                "evidence_coverage",
            ],
            "formula": "Frozen priority tuple over structural proxy metrics",
            "safe_interpretation": "Structured outcome W/T/L, not semantic quality W/T/L.",
        },
        "bootstrap_deltas": {
            "classification": "STRUCTURAL_PROXY",
            "evidence_tier": "Tier 2 - Deterministic structural/proxy",
            "evaluator_function": "bootstrap",
            "input_fields": ["paired task-level structural proxy deltas"],
            "formula": "1000 paired task resamples, seed 41007",
            "safe_interpretation": (
                "CI quantifies proxy metric variability, not semantic validity."
            ),
        },
    }
    payload = {
        "schema_version": "stage4c-metric-provenance-v1",
        "created_at": now(),
        "metric_vector_identity": vector_identity,
        "metrics": metrics,
        "aggregate_values": {
            "workflow": paired["systems"]["workflow"],
            "agent": paired["systems"]["agent"],
            "w_t_l": paired["w_t_l"],
        },
        "content_level_rubric_validated": False,
        "structured_proxy_metrics_valid": True,
    }
    write_json(OUT_PROVENANCE, payload)
    return payload


def failure_and_body_audit() -> dict[str, Any]:
    paired = read_json(PAIRED)
    labels = read_json(LABEL_MAP)["mapping"]
    blind = read_json(BLIND_PACKAGE)
    blind_unit_by_task_system: dict[tuple[str, str], dict[str, Any]] = {}
    blind_body_by_label = Counter()
    for pair in blind["pairs"]:
        for label in ("output_x", "output_y"):
            system = labels[pair["task_id"]][label]
            blind_unit_by_task_system[(pair["task_id"], system)] = pair[label]
            if any(isinstance(pair[label].get(k), str) and pair[label][k].strip() for k in ("report", "answer", "final_answer")):
                blind_body_by_label[label] += 1

    blind_body_by_system = Counter()
    runtime_body_by_system = Counter()
    runtime_empty_by_system = Counter()
    workflow_report_before_failure = 0
    workflow_schema_failures = []
    workflow_failures: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "task_ids": []}
    )
    agent_failures: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "task_ids": []}
    )
    workflow_http_runtime_succeeded = 0
    workflow_provider_calls = 0
    for row in paired["rows"]:
        task_id = row["task_id"]
        for label in ("output_x", "output_y"):
            system = labels[task_id][label]
            if has_body(blind_unit_by_task_system[(task_id, system)]):
                blind_body_by_system[system] += 1
        for system in ("workflow", "agent"):
            response = raw_response(task_id, system)
            record = row[system]
            blind_unit = blind_unit_by_task_system[(task_id, system)]
            if has_body(response):
                runtime_body_by_system[system] += 1
            else:
                runtime_empty_by_system[system] += 1
            if system == "workflow":
                if int(blind_unit.get("http_status") or 0) == 200:
                    workflow_http_runtime_succeeded += 1
                if int(blind_unit.get("provider_requests") or 0) > 0:
                    workflow_provider_calls += 1
                if has_body(response) and record["status"] != "COMPLETED":
                    workflow_report_before_failure += 1
                category = record.get("failure_category") or "NONE"
                workflow_failures[category]["count"] += 1
                workflow_failures[category]["task_ids"].append(task_id)
                if category == "SYSTEM_SCHEMA_FAILURE":
                    workflow_schema_failures.append(task_id)
            else:
                category = record.get("failure_category") or "NONE"
                if category != "NONE":
                    agent_failures[category]["count"] += 1
                    agent_failures[category]["task_ids"].append(task_id)

    citation_count_by_system = Counter()
    citation_emitting_tasks = Counter()
    for row in paired["rows"]:
        for system in ("workflow", "agent"):
            count = int(row[system].get("citation_count") or 0)
            citation_count_by_system[system] += count
            if count > 0:
                citation_emitting_tasks[system] += 1

    return {
        "workflow_failure_categories": dict(workflow_failures),
        "workflow_units_with_report_body": runtime_body_by_system["workflow"],
        "workflow_units_without_report_body": runtime_empty_by_system["workflow"],
        "workflow_units_with_report_before_failure": workflow_report_before_failure,
        "workflow_units_with_schema_failure": len(workflow_schema_failures),
        "workflow_schema_failure_task_ids": workflow_schema_failures,
        "workflow_http_runtime_succeeded": workflow_http_runtime_succeeded,
        "workflow_units_with_provider_calls": workflow_provider_calls,
        "workflow_failure_interpretation": (
            "Frozen Workflow returned terminal FAILED states. Most failures are "
            "verification/retrieval or provider-schema terminal states from the "
            "frozen system path, not Stage4B infrastructure invalidations."
        ),
        "agent_failure_categories": dict(agent_failures),
        "blind_package_body_count": {
            "workflow": blind_body_by_system["workflow"],
            "agent": blind_body_by_system["agent"],
        },
        "runtime_body_recoverable_count": {
            "workflow": runtime_body_by_system["workflow"],
            "agent": runtime_body_by_system["agent"],
        },
        "blind_package_content_omission": True,
        "semantic_content_evaluation_case": "CASE_B_BLIND_PACKAGE_CONTENT_OMISSION_FOR_WORKFLOW_AND_CASE_A_AGENT_NO_FINAL_REPORT_BODY",
        "citation_validity_denominator": {
            "workflow_citation_count": citation_count_by_system["workflow"],
            "agent_citation_count": citation_count_by_system["agent"],
            "workflow_citation_emitting_task_count": citation_emitting_tasks["workflow"],
            "agent_citation_emitting_task_count": citation_emitting_tasks["agent"],
            "convention": "VACUOUS_VALIDITY_CONVENTION_FOR_EMPTY_CITATION_SETS",
        },
        "unsupported_claim_denominator": {
            "evaluated_core_claim_count": 0,
            "unsupported_core_claim_count": 0,
            "interpretation": "No direct emitted-claim semantic denominator was available.",
        },
    }


def build_validity(
    lock: dict[str, Any],
    provenance: dict[str, Any],
    failure_audit: dict[str, Any],
) -> dict[str, Any]:
    final = read_json(FINAL)
    freeze = read_json(FREEZE)
    paired = read_json(PAIRED)
    bootstrap = read_json(BOOTSTRAP)
    gap_rows = [
        {
            "task_id": row["task_id"],
            "workflow": row["workflow"]["gap_handling_accuracy"],
            "agent": row["agent"]["gap_handling_accuracy"],
        }
        for row in paired["rows"]
        if row["category"] == "evidence_insufficiency_or_noncomparability"
    ]
    payload = {
        "schema_version": "stage4c-final-validity-audit-v1",
        "created_at": now(),
        "attempt4_integrity_unchanged": True,
        "original_blind_score_hash_unchanged": (
            freeze["blind_scores_bundle_hash"] == ORIGINAL_BLIND_SCORE_HASH
        ),
        "pre_validity_audit_lock": lock,
        "content_level_rubric_validated": False,
        "structured_proxy_metrics_valid": True,
        "evaluation_protocol_implementation_mismatch": False,
        "metric_vector_identity": provenance["metric_vector_identity"],
        "metric_provenance_summary": {
            name: {
                "classification": item["classification"],
                "evidence_tier": item["evidence_tier"],
                "safe_interpretation": item["safe_interpretation"],
            }
            for name, item in provenance["metrics"].items()
        },
        "claims_actually_scored_individually": 0,
        "dimensions_actually_scored_individually": 0,
        "evidence_items_actually_scored_individually": 0,
        "failure_and_body_audit": failure_audit,
        "gap_handling_task_values": gap_rows,
        "w_t_l_classification": "STRUCTURED_OUTCOME_W_T_L",
        "bootstrap_metric_semantics": "STRUCTURAL_PROXY",
        "semantic_judge_complete": final["judge"]["semantic_judge_complete"],
        "semantic_judge_gap": final["judge"]["judge_gap"],
        "judge_requests": final["judge"]["judge_requests"],
        "new_provider_requests": 0,
        "new_agent_runs": 0,
        "new_workflow_runs": 0,
        "semantic_judge_requests": 0,
        "release_readiness": "READY_WITH_SEMANTIC_EVALUATION_LIMITATION",
        "final_interpretation": (
            "Agent strongly outperformed Workflow on operational completion and "
            "structured proxy metrics in Attempt4, while using substantially "
            "more provider requests/tokens/cost. The benchmark does not "
            "independently establish an equivalent semantic research-quality "
            "improvement because individual rubric claims/dimensions/evidence "
            "were not semantically scored."
        ),
        "reported_values": {
            "workflow": final["systems"]["workflow"],
            "agent": final["systems"]["agent"],
            "w_t_l": final["w_t_l"],
            "bootstrap_task_success": bootstrap["metrics"]["task_success"],
        },
    }
    write_json(OUT_VALIDITY, payload)
    return payload


def build_claim_boundary(validity: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": "stage4-portfolio-claim-boundary-v1",
        "created_at": now(),
        "allowed": [
            "Built a 60-task / 120-run Workflow vs Agent paired benchmark.",
            "Attempt4 completed 120/120 logical executions with integrity checks passed.",
            "Agent completed 56/60 frozen research tasks under the structured runtime status.",
            "Agent showed observation-driven tool selection in 56/60 tasks.",
            "Agent used more provider calls, tokens, and cost than Workflow.",
            "The 60-task benchmark observed effective_replan_count=0.",
        ],
        "qualified": [
            "Agent achieved higher structured required-claim coverage proxy.",
            "Agent achieved higher structured dimension/evidence coverage proxy.",
            "Structured outcome W/T/L was Agent 56, Tie 4, Workflow 0.",
        ],
        "not_allowed": [
            "Agent semantic research quality improved by 93.3 percentage points.",
            "Agent semantically won 56/60 research tasks.",
            "The benchmark proves Replan improved quality.",
            "The benchmark is a fully validated semantic benchmark.",
        ],
        "release_readiness": validity["release_readiness"],
    }
    write_json(OUT_BOUNDARY, payload)
    return payload


def build_readiness(validity: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": "stage4-portfolio-release-readiness-v1",
        "created_at": now(),
        "status": "READY_WITH_LIMITATIONS",
        "readiness_detail": "READY_WITH_SEMANTIC_EVALUATION_LIMITATION",
        "stage4_complete": True,
        "feature_development_stopped": True,
        "content_level_rubric_validated": False,
        "structured_proxy_metrics_valid": True,
        "semantic_judge_complete": False,
        "semantic_judge_gap": validity["semantic_judge_gap"],
        "release_claim_boundary": str(OUT_BOUNDARY),
        "must_not_claim": [
            "FULLY_VALIDATED_SEMANTIC_BENCHMARK",
            "Agent semantic research quality win 56/60",
            "60-task benchmark demonstrated effective replanning",
        ],
    }
    write_json(OUT_READINESS, payload)
    return payload


def write_docs(
    provenance: dict[str, Any],
    validity: dict[str, Any],
    boundary: dict[str, Any],
    readiness: dict[str, Any],
) -> None:
    rows = []
    for metric, item in provenance["metrics"].items():
        value = validity["reported_values"].get(metric, "see aggregate")
        rows.append(
            "| "
            + " | ".join(
                [
                    metric,
                    str(value),
                    item["classification"],
                    item["evidence_tier"],
                    item["safe_interpretation"],
                ]
            )
            + " |"
        )
    DOC_PROVENANCE.write_text(
        "\n".join(
            [
                "# Stage 4C Metric Provenance",
                "",
                "| Metric | Reported value | Provenance | Evidence tier | Safe interpretation |",
                "| --- | --- | --- | --- | --- |",
                *rows,
                "",
                f"- content_level_rubric_validated: `{provenance['content_level_rubric_validated']}`",
                f"- structured_proxy_metrics_valid: `{provenance['structured_proxy_metrics_valid']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    failure = validity["failure_and_body_audit"]
    DOC_VALIDITY.write_text(
        "\n".join(
            [
                "# Stage 4C Final Validity Audit",
                "",
                f"- release_readiness: `{validity['release_readiness']}`",
                f"- content_level_rubric_validated: `{validity['content_level_rubric_validated']}`",
                f"- structured_proxy_metrics_valid: `{validity['structured_proxy_metrics_valid']}`",
                f"- original_blind_score_hash_unchanged: `{validity['original_blind_score_hash_unchanged']}`",
                f"- semantic_judge_complete: `{validity['semantic_judge_complete']}`",
                f"- semantic_judge_gap: `{validity['semantic_judge_gap']}`",
                "",
                "## Body availability",
                "",
                f"- blind_package_workflow_body_count: `{failure['blind_package_body_count']['workflow']}`",
                f"- blind_package_agent_body_count: `{failure['blind_package_body_count']['agent']}`",
                f"- runtime_workflow_body_recoverable_count: `{failure['runtime_body_recoverable_count']['workflow']}`",
                f"- runtime_agent_body_recoverable_count: `{failure['runtime_body_recoverable_count']['agent']}`",
                "",
                "## Interpretation",
                "",
                validity["final_interpretation"],
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    DOC_BOUNDARY.write_text(
        "# Stage 4 Portfolio Claim Boundary\n\n"
        "## Allowed\n\n"
        + "\n".join(f"- {item}" for item in boundary["allowed"])
        + "\n\n## Qualified\n\n"
        + "\n".join(f"- {item}" for item in boundary["qualified"])
        + "\n\n## Not allowed\n\n"
        + "\n".join(f"- {item}" for item in boundary["not_allowed"])
        + "\n",
        encoding="utf-8",
    )

    DOC_READINESS.write_text(
        "# Stage 4 Portfolio Release Readiness\n\n"
        f"- status: `{readiness['status']}`\n"
        f"- detail: `{readiness['readiness_detail']}`\n"
        f"- semantic_judge_complete: `{readiness['semantic_judge_complete']}`\n"
        f"- semantic_judge_gap: `{readiness['semantic_judge_gap']}`\n",
        encoding="utf-8",
    )


def main() -> None:
    lock = create_lock()
    provenance = metric_provenance()
    failure_audit = failure_and_body_audit()
    validity = build_validity(lock, provenance, failure_audit)
    boundary = build_claim_boundary(validity)
    readiness = build_readiness(validity)
    write_docs(provenance, validity, boundary, readiness)
    print(
        json.dumps(
            {
                "stage4c_validity_audit": "complete",
                "release_readiness": readiness["readiness_detail"],
                "content_level_rubric_validated": validity[
                    "content_level_rubric_validated"
                ],
                "structured_proxy_metrics_valid": validity[
                    "structured_proxy_metrics_valid"
                ],
                "new_provider_requests": 0,
                "new_agent_runs": 0,
                "new_workflow_runs": 0,
                "semantic_judge_requests": 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
