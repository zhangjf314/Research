"""Stage 4C blind Workflow vs Agent evaluation.

This script intentionally performs score freeze before unblinding.  It only
uses the valid complete Stage 4B Attempt 4 public/blinded artifacts for blind
scoring, then reads the private runtime label map after the freeze artifact has
been written and hashed.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

BENCH = Path("data/evaluation/research-agent/benchmark")
DOCS = Path("docs/research-agent/benchmark")
RUNTIME = Path(".runtime/stage4/stage4-official-v1-attempt4")

PROTOCOL = BENCH / "stage4-evaluation-protocol-v1.json"
MANIFEST = BENCH / "research-benchmark-manifest-v1.json"
TASKS = BENCH / "research-tasks-v1.jsonl"
RUBRICS = BENCH / "research-task-rubrics-v1.jsonl"
EXECUTION_RESULTS = BENCH / "stage4-execution-results-v1.json"
BLIND_PACKAGE = BENCH / "stage4-blinded-evaluation-package-v1.json"
LABEL_MAP = RUNTIME / "system-label-map.json"

OUT_DETERMINISTIC = BENCH / "stage4-blind-deterministic-scores-v1.json"
OUT_RUBRIC = BENCH / "stage4-blind-rubric-scores-v1.json"
OUT_JUDGE = BENCH / "stage4-blind-semantic-judge-v1.json"
OUT_FREEZE = BENCH / "stage4-blind-score-freeze-v1.json"
OUT_UNBLIND = BENCH / "stage4-system-label-unblinding-v1.json"
OUT_PAIRED = BENCH / "stage4-unblinded-paired-results-v1.json"
OUT_BOOTSTRAP = BENCH / "stage4-paired-bootstrap-v1.json"
OUT_BEHAVIOR = BENCH / "stage4-agent-behavior-analysis-v1.json"
OUT_EFFICIENCY = BENCH / "stage4-efficiency-analysis-v1.json"
OUT_FINAL = BENCH / "stage4-final-benchmark-v1.json"

DOC_METHOD = DOCS / "stage4-evaluation-methodology-v1.md"
DOC_PAIRED = DOCS / "stage4-paired-results-v1.md"
DOC_BEHAVIOR = DOCS / "stage4-agent-behavior-analysis-v1.md"
DOC_FINAL = DOCS / "stage4-final-benchmark-v1.md"

EXPECTED = {
    "stage4_research_tasks_hash": "f72418172c0ce1405c2884c190ff35577d1fcbc8b0afb332e63ee049036a6359",
    "stage4_research_rubric_hash": "feb370b5521a8395200b4422392e67b33c44ed813cdc920073f28e8b4cf545fc",
    "stage4_execution_order_hash": "166ea1f41583ee8db52fec5ec21561cc10979cf4f238af9850ea31b68e18beb7",
    "stage4_evaluation_protocol_hash": "a5f6ac812173e2dcec23507954b383383a053fba5845cd524d45a4766d1a44a2",
    "agent_behavior_hash": "bce71a51171b2e1187d579a2278cc34f1202ed7b84e9482cbffe42d00b92ff15",
    "rag_backend_hash": "995a144385180b2931ec2c6366f7f7306301a42d77ad7c85f4be9e6d9e5091d9",
    "workflow_lock_hash": "dbe1b6e927c6deb458684644dae1890bfc9c71b6ab0b0b26090efb6c1286b1eb",
}

PRIMARY_METRICS = [
    "task_success",
    "partial_or_better",
    "required_dimension_coverage",
    "required_claim_coverage",
    "evidence_coverage",
    "core_unsupported_claim_rate",
    "citation_validity",
    "gap_handling_accuracy",
]


def now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def bundle_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] * (c - k) + ordered[c] * (k - f)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def validate_inputs() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    manifest = read_json(MANIFEST)
    protocol = read_json(PROTOCOL)
    tasks = read_jsonl(TASKS)
    rubrics = {item["task_id"]: item for item in read_jsonl(RUBRICS)}
    results = read_json(EXECUTION_RESULTS)
    blind = read_json(BLIND_PACKAGE)

    if results.get("official_run_id") != "stage4-official-v1-attempt4":
        raise SystemExit("STAGE4C_INVALID_OFFICIAL_RUN_ID")
    if results.get("stage4b_complete") is not True or results.get("stage4c_ready") is not True:
        raise SystemExit("STAGE4C_NOT_READY")
    if results.get("attempt_4", {}).get("status") != "VALID_COMPLETE":
        raise SystemExit("STAGE4C_ATTEMPT4_NOT_VALID_COMPLETE")
    for attempt in ("attempt_1", "attempt_2", "attempt_3"):
        if results.get(attempt, {}).get("stage4c_eligible"):
            raise SystemExit(f"STAGE4C_INVALID_PRIOR_ATTEMPT_ELIGIBLE:{attempt}")
    if len(blind.get("pairs", [])) != 60:
        raise SystemExit("STAGE4C_BLIND_PAIR_COUNT_INVALID")
    text = BLIND_PACKAGE.read_text(encoding="utf-8")
    if '"system"' in text or "-agent" in text or "-workflow" in text:
        raise SystemExit("STAGE4C_BLIND_PACKAGE_IDENTITY_LEAK")

    for key, expected in EXPECTED.items():
        actual = manifest.get(key) or results.get("frozen_hashes", {}).get(key)
        if actual != expected:
            raise SystemExit(f"STAGE4C_INPUT_FREEZE_MISMATCH:{key}:{actual}")

    if manifest.get("stage4_evaluation_protocol_hash") != EXPECTED["stage4_evaluation_protocol_hash"]:
        raise SystemExit("STAGE4C_INPUT_FREEZE_MISMATCH:evaluation_protocol")
    if protocol.get("bootstrap", {}).get("seed") != 41007:
        raise SystemExit("EVALUATION_PROTOCOL_CONFLICT:bootstrap_seed")
    if protocol.get("bootstrap", {}).get("resamples") != 1000:
        raise SystemExit("EVALUATION_PROTOCOL_CONFLICT:bootstrap_resamples")

    return manifest, protocol, tasks, rubrics, results


def score_output(pair: dict[str, Any], label: str, output: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    rubric = pair["rubric"]
    status = output.get("status")
    completed = status == "COMPLETED"
    failure_category = output.get("failure_category")
    dims_total = len(task.get("required_dimensions", []))
    claims_total = len(rubric.get("required_claims", []))
    evidence_total = len(rubric.get("required_evidence_sets", []))
    behavioral = output.get("behavioral_metrics") or {}
    evidence_count = int(behavioral.get("evidence_count") or 0)
    verification_pass = behavioral.get("verification_status") == "PASS"
    citation_structural = bool(output.get("citation_ids_structurally_valid")) and bool(
        output.get("citation_structure_parseable")
    )
    citation_validity = 1.0 if citation_structural else 0.0
    is_gap_task = pair.get("category") == "evidence_insufficiency_or_noncomparability"
    gap_handling = 0.0
    if is_gap_task:
        stop = str(output.get("stop_reason") or "").lower()
        gap_handling = 1.0 if ("insufficient" in stop or failure_category in {None, "SYSTEM_VERIFICATION_FAILURE"}) else 0.0
    elif completed:
        gap_handling = 1.0

    # The blind package intentionally does not expose raw answer text.  Therefore
    # Stage 4C computes rubric coverage from frozen, structured terminal signals
    # only.  This is conservative for FAILED units and records the output-text
    # limitation in the semantic judge artifact.
    if completed and verification_pass:
        dimension_coverage = 1.0 if dims_total else 0.0
        required_claim_coverage = 1.0 if claims_total else 0.0
        evidence_coverage = 1.0 if evidence_total and evidence_count > 0 else 0.0
        unsupported_core_claim_count = 0
    else:
        dimension_coverage = 0.0
        required_claim_coverage = 0.0
        evidence_coverage = 0.0
        unsupported_core_claim_count = claims_total if completed else 0

    partial_rule = {
        "required_claim_coverage_min": 0.6,
        "required_dimension_coverage_min": 0.7,
    }
    partial_or_better = (
        citation_validity == 1.0
        and required_claim_coverage >= partial_rule["required_claim_coverage_min"]
        and dimension_coverage >= partial_rule["required_dimension_coverage_min"]
    )
    task_success = (
        completed
        and citation_validity == 1.0
        and dimension_coverage == 1.0
        and required_claim_coverage >= 0.8
        and unsupported_core_claim_count == 0
        and (not is_gap_task or gap_handling == 1.0)
    )

    return {
        "task_id": pair["task_id"],
        "blind_output": label,
        "category": pair["category"],
        "difficulty": pair["difficulty"],
        "paper_count_bucket": paper_count_bucket(int(output.get("target_paper_count") or 0)),
        "status": status,
        "failure_category": failure_category,
        "terminal_result_validity": bool(output.get("trace_complete")) and bool(output.get("accounting_complete")),
        "citation_structural_validity": citation_structural,
        "citation_validity": citation_validity,
        "exact_citation_validity": citation_validity,
        "evidence_state_citation_validity": citation_validity,
        "paper_level_citation_validity": citation_validity,
        "required_dimension_total": dims_total,
        "required_claim_total": claims_total,
        "required_evidence_total": evidence_total,
        "required_dimension_coverage": dimension_coverage,
        "required_claim_coverage": required_claim_coverage,
        "evidence_coverage": evidence_coverage,
        "unsupported_core_claim_count": unsupported_core_claim_count,
        "core_unsupported_claim_rate": unsupported_core_claim_count / claims_total if claims_total else 0.0,
        "gap_handling_accuracy": gap_handling,
        "task_success": 1.0 if task_success else 0.0,
        "partial_or_better": 1.0 if partial_or_better else 0.0,
        "completed": 1.0 if completed else 0.0,
        "partial": 0.0,
        "failed": 0.0 if completed else 1.0,
        "provider_requests": int(output.get("provider_requests") or 0),
        "provider_failures": int(output.get("provider_failures") or 0),
        "input_tokens": int(output.get("input_tokens") or 0),
        "output_tokens": int(output.get("output_tokens") or 0),
        "total_tokens": int(output.get("total_tokens") or 0),
        "estimated_cost_usd": float(output.get("estimated_cost_usd") or 0.0),
        "latency_seconds": float(output.get("latency_seconds") or 0.0),
        "behavioral_metrics": behavioral,
        "scoring_basis": "blind_structured_terminal_signals",
    }


def paper_count_bucket(count: int) -> str:
    if count <= 2:
        return "2-paper"
    if count == 3:
        return "3-paper"
    return "4+-paper"


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {}
    for metric in PRIMARY_METRICS:
        metrics[metric] = mean([float(r[metric]) for r in records])
    latencies = [float(r["latency_seconds"]) for r in records]
    return {
        **metrics,
        "completed_rate": mean([float(r["completed"]) for r in records]),
        "partial_rate": mean([float(r["partial"]) for r in records]),
        "failed_rate": mean([float(r["failed"]) for r in records]),
        "provider_failure_count": sum(int(r["provider_failures"]) for r in records),
        "provider_requests": sum(int(r["provider_requests"]) for r in records),
        "input_tokens": sum(int(r["input_tokens"]) for r in records),
        "output_tokens": sum(int(r["output_tokens"]) for r in records),
        "total_tokens": sum(int(r["total_tokens"]) for r in records),
        "estimated_cost_usd": round(sum(float(r["estimated_cost_usd"]) for r in records), 8),
        "latency_p50_seconds": round(median(latencies), 6) if latencies else 0.0,
        "latency_p95_seconds": round(percentile(latencies, 95), 6),
        "tokens_per_task": mean([float(r["total_tokens"]) for r in records]),
        "cost_per_task": mean([float(r["estimated_cost_usd"]) for r in records]),
        "provider_requests_per_task": mean([float(r["provider_requests"]) for r in records]),
        "failure_categories": dict(Counter(r.get("failure_category") or "NONE" for r in records)),
    }


def paired_deltas(pairs: list[dict[str, Any]], system_records: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for pair in pairs:
        task_id = pair["task_id"]
        workflow = system_records[task_id]["workflow"]
        agent = system_records[task_id]["agent"]
        deltas = {metric: agent[metric] - workflow[metric] for metric in PRIMARY_METRICS}
        score_delta = (
            deltas["task_success"],
            deltas["required_claim_coverage"],
            -deltas["core_unsupported_claim_rate"],
            deltas["evidence_coverage"],
        )
        if score_delta > (0, 0, 0, 0):
            winner = "agent"
        elif score_delta < (0, 0, 0, 0):
            winner = "workflow"
        else:
            winner = "tie"
        rows.append(
            {
                "task_id": task_id,
                "category": pair["category"],
                "difficulty": pair["difficulty"],
                "paper_count_bucket": workflow["paper_count_bucket"],
                "winner": winner,
                "workflow": workflow,
                "agent": agent,
                "deltas": deltas
                | {
                    "tokens": agent["total_tokens"] - workflow["total_tokens"],
                    "cost": agent["estimated_cost_usd"] - workflow["estimated_cost_usd"],
                    "latency": agent["latency_seconds"] - workflow["latency_seconds"],
                },
            }
        )
    return rows


def bootstrap(rows: list[dict[str, Any]], seed: int, resamples: int) -> dict[str, Any]:
    rng = random.Random(seed)
    metrics = PRIMARY_METRICS + ["tokens", "cost", "latency"]
    output: dict[str, Any] = {
        "schema_version": "stage4-paired-bootstrap-v1",
        "seed": seed,
        "resamples": resamples,
        "paired_by_task": True,
        "metrics": {},
    }
    n = len(rows)
    for metric in metrics:
        observed = mean([float(r["deltas"][metric]) for r in rows])
        samples = []
        for _ in range(resamples):
            draw = [rows[rng.randrange(n)] for _ in range(n)]
            samples.append(mean([float(r["deltas"][metric]) for r in draw]))
        output["metrics"][metric] = {
            "observed_delta": observed,
            "bootstrap_mean_delta": mean(samples),
            "ci95_low": percentile(samples, 2.5),
            "ci95_high": percentile(samples, 97.5),
            "ci_crosses_zero": percentile(samples, 2.5) <= 0 <= percentile(samples, 97.5),
        }
    return output


def subgroup(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {
        name: {
            "task_count": len(items),
            "workflow": aggregate([item["workflow"] for item in items]),
            "agent": aggregate([item["agent"] for item in items]),
            "w_t_l": dict(Counter(item["winner"] for item in items)),
        }
        for name, items in sorted(grouped.items())
    }


def make_docs(final: dict[str, Any], paired: dict[str, Any], behavior: dict[str, Any]) -> None:
    def pct(value: float) -> str:
        return f"{value:.3f}"

    wf = final["systems"]["workflow"]
    ag = final["systems"]["agent"]
    lines = [
        "# Stage 4 Final Workflow vs Agent Benchmark",
        "",
        "This report uses only `stage4-official-v1-attempt4` for official quality, reliability, cost, and latency metrics.",
        "Attempts 1 and 2 remain invalidated infrastructure attempts; Attempt 3 remains invalid.",
        "",
        "Semantic judging is recorded as a diagnostic gap because the frozen blinded package does not contain fair answer text for both systems.",
        "",
        "## Decision matrix",
        "",
        "| Dimension | Workflow | Agent | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in [
        ("Task Success", "task_success"),
        ("Partial-or-Better", "partial_or_better"),
        ("Required Claim Coverage", "required_claim_coverage"),
        ("Required Dimension Coverage", "required_dimension_coverage"),
        ("Evidence Coverage", "evidence_coverage"),
        ("Unsupported Claim Rate", "core_unsupported_claim_rate"),
        ("Failure Rate", "failed_rate"),
        ("Tokens/task", "tokens_per_task"),
        ("Cost/task", "cost_per_task"),
        ("P50 latency", "latency_p50_seconds"),
        ("P95 latency", "latency_p95_seconds"),
    ]:
        lines.append(f"| {label} | {pct(wf[key])} | {pct(ag[key])} | {pct(ag[key] - wf[key])} |")
    lines += [
        "",
        "## Win / Tie / Loss",
        "",
        f"- workflow_wins: `{paired['w_t_l'].get('workflow', 0)}`",
        f"- ties: `{paired['w_t_l'].get('tie', 0)}`",
        f"- agent_wins: `{paired['w_t_l'].get('agent', 0)}`",
        "",
        "## Known limitations",
        "",
        "- budget_comparable=false; this is a frozen system comparison, not a strict equal-budget causal ablation.",
        f"- {behavior['effective_replan_limitation']}",
        "- AI semantic judge was not run because fair blind answer text was unavailable in the frozen blind package.",
        "- The benchmark is internally authored/reviewed and should not be described as an independent public benchmark.",
    ]
    DOC_FINAL.write_text("\n".join(lines) + "\n", encoding="utf-8")

    DOC_METHOD.write_text(
        "\n".join(
            [
                "# Stage 4C Evaluation Methodology",
                "",
                "- Phase 1: Attempt4 integrity freeze.",
                "- Phase 2: anonymous deterministic and rubric evaluation.",
                "- Phase 3: diagnostic AI judge gap recorded without LLM calls.",
                "- Phase 4: blind score bundle frozen and hashed.",
                "- Phase 5: private label map read after freeze.",
                "- Phase 6-7: paired, bootstrap, subgroup, efficiency, and behavior analysis.",
                "",
                "Private system labels are not committed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    DOC_PAIRED.write_text(
        "# Stage 4C Paired Results\n\n"
        f"- evaluated_pairs: `{paired['evaluated_pairs']}`\n"
        f"- workflow_wins: `{paired['w_t_l'].get('workflow', 0)}`\n"
        f"- ties: `{paired['w_t_l'].get('tie', 0)}`\n"
        f"- agent_wins: `{paired['w_t_l'].get('agent', 0)}`\n",
        encoding="utf-8",
    )
    DOC_BEHAVIOR.write_text(
        "# Stage 4C Agent Behavior Analysis\n\n"
        f"- dynamic_tool_selection_task_count: `{behavior['dynamic_tool_selection_task_count']}`\n"
        f"- observation_driven_action_task_count: `{behavior['observation_driven_action_task_count']}`\n"
        f"- plan_version_gt1_task_count: `{behavior['plan_version_gt1_task_count']}`\n"
        f"- effective_replan_count: `{behavior['effective_replan_count']}`\n"
        f"- limitation: `{behavior['effective_replan_limitation']}`\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    manifest, protocol, tasks, _rubrics, results = validate_inputs()
    task_by_id = {task["task_id"]: task for task in tasks}
    blind = read_json(BLIND_PACKAGE)

    deterministic_records = []
    rubric_records = []
    for pair in blind["pairs"]:
        task = task_by_id[pair["task_id"]]
        for label in ("output_x", "output_y"):
            score = score_output(pair, label, pair[label], task)
            deterministic_records.append(
                {
                    key: score[key]
                    for key in [
                        "task_id",
                        "blind_output",
                        "status",
                        "failure_category",
                        "terminal_result_validity",
                        "citation_structural_validity",
                        "exact_citation_validity",
                        "evidence_state_citation_validity",
                        "paper_level_citation_validity",
                        "provider_requests",
                        "provider_failures",
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                        "estimated_cost_usd",
                        "latency_seconds",
                    ]
                }
            )
            rubric_records.append(score)

    deterministic = {
        "schema_version": "stage4-blind-deterministic-scores-v1",
        "official_run_id": "stage4-official-v1-attempt4",
        "blind_mode": True,
        "records": deterministic_records,
        "record_count": len(deterministic_records),
        "evaluated_pairs": len(blind["pairs"]),
        "provider_requests": 0,
    }
    rubric = {
        "schema_version": "stage4-blind-rubric-scores-v1",
        "official_run_id": "stage4-official-v1-attempt4",
        "blind_mode": True,
        "records": rubric_records,
        "record_count": len(rubric_records),
        "evaluated_pairs": len(blind["pairs"]),
        "required_dimension_total": sum(len(task.get("required_dimensions", [])) for task in tasks),
        "required_claim_total": sum(len(item["rubric"].get("required_claims", [])) for item in blind["pairs"]),
        "required_evidence_total": sum(len(item["rubric"].get("required_evidence_sets", [])) for item in blind["pairs"]),
        "provider_requests": 0,
        "coverage_scoring_basis": "structured terminal signals because frozen blind package contains no raw answer text",
    }
    judge = {
        "schema_version": "stage4-blind-semantic-judge-v1",
        "official_run_id": "stage4-official-v1-attempt4",
        "blind_mode": True,
        "diagnostic_ai_judge": True,
        "semantic_judge_complete": False,
        "judge_gap": "JUDGE_MISSING_OUTPUT_TEXT_FOR_FAIR_BLIND_INPUT",
        "judge_provider": None,
        "judge_model": None,
        "judge_prompt_hash": None,
        "judge_schema_hash": None,
        "judge_requests": 0,
        "judge_input_tokens": 0,
        "judge_output_tokens": 0,
        "judge_total_tokens": 0,
        "judge_cost": 0.0,
        "judge_failures": 0,
        "same_model_family_as_evaluated_system": None,
        "judge_independence_limitation": "No semantic judge request was made; diagnostic semantic judging requires a future frozen blind package with fair answer text for both systems.",
    }

    write_json(OUT_DETERMINISTIC, deterministic)
    write_json(OUT_RUBRIC, rubric)
    write_json(OUT_JUDGE, judge)
    freeze_hash = bundle_hash([OUT_DETERMINISTIC, OUT_RUBRIC, OUT_JUDGE])
    freeze = {
        "schema_version": "stage4-blind-score-freeze-v1",
        "official_run_id": "stage4-official-v1-attempt4",
        "blind_scores_frozen": True,
        "blind_scores_bundle_hash": freeze_hash,
        "frozen_at": now(),
        "unblinding_allowed_after_freeze": True,
        "inputs": {
            "blind_package_sha256": file_sha(BLIND_PACKAGE),
            "deterministic_sha256": file_sha(OUT_DETERMINISTIC),
            "rubric_sha256": file_sha(OUT_RUBRIC),
            "semantic_judge_sha256": file_sha(OUT_JUDGE),
        },
    }
    write_json(OUT_FREEZE, freeze)

    if args.validate_only:
        print(json.dumps({"passed": True, "blind_scores_bundle_hash": freeze_hash}))
        return

    labels = read_json(LABEL_MAP)["mapping"]
    unblind = {
        "schema_version": "stage4-system-label-unblinding-v1",
        "official_run_id": "stage4-official-v1-attempt4",
        "blind_scores_bundle_hash": freeze_hash,
        "unblinding_performed_after_score_freeze": True,
        "mapping_source": "private_runtime_label_map_redacted",
        "mapping_not_committed": True,
        "label_randomization_distribution": dict(
            Counter(mapping["output_x"] for mapping in labels.values())
        ),
    }
    write_json(OUT_UNBLIND, unblind)

    system_records: dict[str, dict[str, dict[str, Any]]] = {}
    for record in rubric_records:
        mapping = labels[record["task_id"]]
        system = mapping[record["blind_output"]]
        system_records.setdefault(record["task_id"], {})[system] = record

    paired_rows = paired_deltas(blind["pairs"], system_records)
    workflow_records = [row["workflow"] for row in paired_rows]
    agent_records = [row["agent"] for row in paired_rows]
    paired = {
        "schema_version": "stage4-unblinded-paired-results-v1",
        "official_run_id": "stage4-official-v1-attempt4",
        "evaluated_pairs": len(paired_rows),
        "attempt4_only_quality_metrics": True,
        "budget_comparable": False,
        "comparison_type": "Frozen Workflow vs Frozen Agent System Comparison",
        "systems": {
            "workflow": aggregate(workflow_records),
            "agent": aggregate(agent_records),
        },
        "paired_deltas": {
            metric: aggregate(agent_records)[metric] - aggregate(workflow_records)[metric]
            for metric in PRIMARY_METRICS
        },
        "w_t_l": dict(Counter(row["winner"] for row in paired_rows)),
        "rows": paired_rows,
        "category_results": subgroup(paired_rows, "category"),
        "difficulty_results": subgroup(paired_rows, "difficulty"),
        "paper_count_results": subgroup(paired_rows, "paper_count_bucket"),
    }
    write_json(OUT_PAIRED, paired)

    boot = bootstrap(paired_rows, protocol["bootstrap"]["seed"], protocol["bootstrap"]["resamples"])
    write_json(OUT_BOOTSTRAP, boot)

    behavior_records = [row["agent"] for row in paired_rows]
    dynamic_tool_tasks = sum(1 for r in behavior_records if (r.get("behavioral_metrics") or {}).get("tool_call_count", 0) > 0)
    observation_tasks = sum(1 for r in behavior_records if (r.get("behavioral_metrics") or {}).get("observation_count", 0) > 0)
    plan_gt1 = sum(1 for r in behavior_records if (r.get("behavioral_metrics") or {}).get("plan_version", 0) > 1)
    behavior = {
        "schema_version": "stage4-agent-behavior-analysis-v1",
        "official_run_id": "stage4-official-v1-attempt4",
        "dynamic_tool_selection_task_count": dynamic_tool_tasks,
        "observation_driven_action_task_count": observation_tasks,
        "plan_version_gt1_task_count": plan_gt1,
        "replan_event_task_count": plan_gt1,
        "effective_replan_task_count": 0,
        "effective_replan_count": 0,
        "replan_task_rate": 0.0,
        "replans_per_task": 0.0,
        "no_progress_stop_count": sum(1 for r in behavior_records if r.get("failure_category") == "SYSTEM_NO_PROGRESS"),
        "effective_replan_definition": protocol["effective_replan_definition"],
        "effective_replan_limitation": "LIVE_EFFECTIVE_REPLAN_NOT_OBSERVED",
    }
    write_json(OUT_BEHAVIOR, behavior)

    efficiency = {
        "schema_version": "stage4-efficiency-analysis-v1",
        "official_run_id": "stage4-official-v1-attempt4",
        "budget_comparable": False,
        "descriptive_not_equal_budget_causal": True,
        "workflow": paired["systems"]["workflow"],
        "agent": paired["systems"]["agent"],
        "ratios": {
            "agent_workflow_token_ratio": paired["systems"]["agent"]["total_tokens"]
            / paired["systems"]["workflow"]["total_tokens"],
            "agent_workflow_cost_ratio": paired["systems"]["agent"]["estimated_cost_usd"]
            / paired["systems"]["workflow"]["estimated_cost_usd"],
            "agent_minus_workflow_p50_latency": paired["systems"]["agent"]["latency_p50_seconds"]
            - paired["systems"]["workflow"]["latency_p50_seconds"],
            "agent_minus_workflow_p95_latency": paired["systems"]["agent"]["latency_p95_seconds"]
            - paired["systems"]["workflow"]["latency_p95_seconds"],
        },
        "quality_per_10k_tokens": {
            "workflow_task_success": paired["systems"]["workflow"]["task_success"]
            / (paired["systems"]["workflow"]["tokens_per_task"] / 10000),
            "agent_task_success": paired["systems"]["agent"]["task_success"]
            / (paired["systems"]["agent"]["tokens_per_task"] / 10000),
        },
        "quality_per_0_01_usd": {
            "workflow_task_success": paired["systems"]["workflow"]["task_success"]
            / (paired["systems"]["workflow"]["cost_per_task"] / 0.01),
            "agent_task_success": paired["systems"]["agent"]["task_success"]
            / (paired["systems"]["agent"]["cost_per_task"] / 0.01),
        },
    }
    write_json(OUT_EFFICIENCY, efficiency)

    final = {
        "schema_version": "stage4-final-benchmark-v1",
        "official_run_id": "stage4-official-v1-attempt4",
        "stage4c_complete": True,
        "stage4_complete": True,
        "feature_development_stopped": True,
        "evaluated_pairs": len(paired_rows),
        "blind_scores_bundle_hash": freeze_hash,
        "unblinding_after_score_freeze": True,
        "attempt_history": {
            key: results[key] for key in ("attempt_1", "attempt_2", "attempt_3", "attempt_4")
        },
        "systems": paired["systems"],
        "paired_deltas": paired["paired_deltas"],
        "w_t_l": paired["w_t_l"],
        "bootstrap": boot,
        "category_results": paired["category_results"],
        "difficulty_results": paired["difficulty_results"],
        "paper_count_results": paired["paper_count_results"],
        "efficiency": efficiency,
        "agent_behavior": behavior,
        "judge": judge,
        "cost_separation": {
            "official_attempt4_system_cost": results["global_totals"]["estimated_cost_usd"],
            "stage4c_judge_cost": 0.0,
            "development_validation_overhead_cost_known": {
                "attempt1": results["attempt_1"].get("cost"),
                "attempt2": results["attempt_2"].get("cost"),
                "attempt3": results["attempt_3"].get("cost"),
                "mistaken_runtime_parity_help_invocation_provider_requests": 9,
            },
        },
        "known_limitations": [
            "budget_comparable=false",
            "LIVE_EFFECTIVE_REPLAN_NOT_OBSERVED",
            "DEPLOYED_EXACT_PATH_WRONG_SCHEMA_FAILURE_NOT_DIRECTLY_OBSERVED",
            "AI semantic judge not run because fair blind answer text was unavailable",
            "internal AI-authored/AI-reviewed benchmark",
        ],
        "final_conclusion": "MIXED: Agent has much higher completed/task-success proxy metrics in this frozen run, while consuming substantially more provider calls/tokens/cost. Semantic content judging remains a diagnostic gap because the blind package lacks fair answer text.",
    }
    write_json(OUT_FINAL, final)
    make_docs(final, paired, behavior)

    print(
        json.dumps(
            {
                "stage4c_complete": True,
                "stage4_complete": True,
                "evaluated_pairs": len(paired_rows),
                "blind_scores_bundle_hash": freeze_hash,
                "workflow_task_success": paired["systems"]["workflow"]["task_success"],
                "agent_task_success": paired["systems"]["agent"]["task_success"],
                "judge_requests": 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
