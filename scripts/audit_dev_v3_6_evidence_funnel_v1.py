"""Build Dev v3.6 required-claim evidence funnel diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "evaluation"
DOCS = ROOT / "docs"
RUN_ROOT = DATA / "evidence-qa-dev-v3-6" / "runs"
SUMMARY = DATA / "evidence-qa-dev-v3-6.json"
GOLD = DATA / "claim-evidence-gold-dev-v1.jsonl"

OUT_JSONL = DATA / "dev-v3-6-evidence-funnel-v1.jsonl"
OUT_JSON = DATA / "dev-v3-6-evidence-funnel-v1.json"
OUT_CSV = DATA / "dev-v3-6-evidence-funnel-v1.csv"
OUT_DOC = DOCS / "dev-v3-6-evidence-funnel-v1.md"
METRICS_JSON = DATA / "dev-v3-6-evidence-funnel-metrics-v1.json"
METRICS_DOC = DOCS / "dev-v3-6-evidence-funnel-metrics-v1.md"


def canonical_hash(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def relation_key(row: dict[str, Any]) -> str:
    return f"{row['paper_id']}|{row['page']}|{row['block_id']}"


def load_gold() -> dict[str, dict[str, Any]]:
    return {row["required_claim_id"]: row for row in read_jsonl(GOLD) if row["answerable"]}


def relation_sets(record: dict[str, Any]) -> dict[str, set[str]]:
    by_id = {rel["relation_id"]: rel for rel in record["candidate_evidence_relations"]}
    core_ids = set(record.get("approved_core_relations", []))
    supporting_ids = set(record.get("approved_supporting_relations", []))
    equivalent_ids = set(record.get("equivalent_non_gold_relations", []))
    return {
        "core": {relation_key(by_id[rel_id]) for rel_id in core_ids if rel_id in by_id},
        "supporting": {
            relation_key(by_id[rel_id]) for rel_id in supporting_ids if rel_id in by_id
        },
        "equivalent": {
            relation_key(by_id[rel_id]) for rel_id in equivalent_ids if rel_id in by_id
        },
        "core_relation_ids": core_ids,
        "supporting_relation_ids": supporting_ids,
        "equivalent_relation_ids": equivalent_ids,
    }


def registry_maps(run_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    registry = read_json(run_dir / "citation-registry.json")
    by_id = {entry["citation_id"]: entry for entry in registry["entries"]}
    key_by_id = {
        entry["citation_id"]: f"{entry['paper_id']}|{entry['page']}|{entry['block_id']}"
        for entry in registry["entries"]
    }
    return by_id, key_by_id


def candidate_rows(run_dir: Path, required_claim_id: str) -> list[dict[str, Any]]:
    local = read_json(run_dir / "candidate-evidence-local.json")
    for row in local["candidate_rows"]:
        if row["required_claim_id"] == required_claim_id:
            return list(row["candidates"])
    return []


def trace_slot(run_dir: Path, required_claim_id: str) -> dict[str, Any]:
    trace = read_json(run_dir / "citation-selection-trace.json")
    for row in trace["slots"]:
        if row["required_claim_id"] == required_claim_id:
            return row
    raise KeyError(required_claim_id)


def final_slot(run_dir: Path, required_claim_id: str) -> dict[str, Any]:
    result = read_json(run_dir / "final-result.json")
    for row in result["final_answer"]["required_claim_results"]:
        if row["required_claim_id"] == required_claim_id:
            return row
    raise KeyError(required_claim_id)


def raw_slot(run_dir: Path, required_claim_id: str) -> dict[str, Any]:
    result = read_json(run_dir / "final-result.json")
    for row in result["raw_model_payload"]["required_claim_results"]:
        if row["required_claim_id"] == required_claim_id:
            return row
    raise KeyError(required_claim_id)


def has_any(keys: set[str], valid: set[str]) -> bool:
    return bool(keys & valid)


def root_cause(
    *,
    valid_exists: bool,
    retrieved_any: bool,
    candidate_any: bool,
    selected_any: bool,
    cited_any: bool,
    core_complete: bool,
    narrowed: bool,
    unsupported: bool,
    cap_blocked: bool,
) -> tuple[str, list[str], str]:
    secondary: list[str] = []
    if not valid_exists:
        return "legacy_metric_only", secondary, "metric_only_legacy_gold_issue"
    if not retrieved_any:
        return "core_evidence_not_retrieved", secondary, "retrieval"
    if not candidate_any:
        return "retrieved_but_candidate_pruned", secondary, "candidate_pruning"
    if not selected_any:
        return "policy_ranked_wrong_primary", secondary, "selection"
    if not cited_any:
        return "selected_but_not_cited", secondary, "selection"
    if cap_blocked:
        secondary.append("citation_cap_blocked_complete_set")
    if narrowed:
        secondary.append("unnecessary_narrowing")
        return "unnecessary_narrowing", secondary, "claim_fallback"
    if unsupported:
        secondary.append("unnecessary_unsupported")
        return "unnecessary_unsupported", secondary, "claim_fallback"
    if not core_complete:
        return "claim_not_supported_by_available_evidence", secondary, "selection"
    return "no_failure", secondary, "no_failure"


def build_rows() -> list[dict[str, Any]]:
    summary = read_json(SUMMARY)
    runs_by_qid = {
        item["question_id"]: item["run_id"]
        for item in summary["attempt_history"]
        if item.get("selected")
    }
    gold = load_gold()
    rows: list[dict[str, Any]] = []
    for required_claim_id, gold_record in sorted(gold.items()):
        qid = gold_record["question_id"]
        run_dir = RUN_ROOT / runs_by_qid[qid]
        registry, key_by_citation = registry_maps(run_dir)
        raw = raw_slot(run_dir, required_claim_id)
        final = final_slot(run_dir, required_claim_id)
        trace = trace_slot(run_dir, required_claim_id)
        candidates = candidate_rows(run_dir, required_claim_id)
        candidate_keys = {
            f"{row['paper_id']}|{row['page']}|{row['block_id']}" for row in candidates
        }
        selected_ids = set(trace["primary_citation_ids"] + trace["supporting_citation_ids"])
        cited_ids = set(final["citation_ids"])
        selected_keys = {key_by_citation[cid] for cid in selected_ids if cid in key_by_citation}
        cited_keys = {key_by_citation[cid] for cid in cited_ids if cid in key_by_citation}
        registry_keys = set(key_by_citation.values())
        sets = relation_sets(gold_record)
        exact = sets["core"] | sets["supporting"]
        any_valid = exact | sets["equivalent"]
        valid_exists = bool(any_valid)
        retrieved_any = has_any(registry_keys, any_valid)
        candidate_any = has_any(candidate_keys, any_valid)
        selected_any = has_any(selected_keys, any_valid)
        cited_any = has_any(cited_keys, any_valid)
        core_complete = bool(sets["core"]) and sets["core"].issubset(cited_keys)
        narrowed = final["status"] == "answered" and bool(trace.get("removed_obligations"))
        unsupported = final["status"] == "unsupported"
        cap_blocked = len(selected_ids) >= 3 and not core_complete
        primary, secondary, stage = root_cause(
            valid_exists=valid_exists,
            retrieved_any=retrieved_any,
            candidate_any=candidate_any,
            selected_any=selected_any,
            cited_any=cited_any,
            core_complete=core_complete,
            narrowed=narrowed,
            unsupported=unsupported,
            cap_blocked=cap_blocked,
        )
        rows.append(
            {
                "question_id": qid,
                "required_claim_id": required_claim_id,
                "required_claim_text": gold_record["required_claim_text"],
                "raw_model_claim_text": raw.get("claim_text"),
                "final_claim_text": final.get("claim_text"),
                "raw_status_shape": "answered_shape"
                if "claim_text" in raw
                else "unsupported_shape",
                "final_status": final["status"],
                "narrowed": narrowed,
                "removed_obligations": trace.get("removed_obligations", []),
                "unsupported_reason": final.get("omission_reason"),
                "candidate_paper_scope": sorted(
                    {entry["paper_id"] for entry in registry.values()}
                ),
                "retrieval_query": gold_record["required_claim_text"],
                "retrieved_candidate_count": len(registry_keys),
                "retrieved_original_count": sum(
                    1 for row in candidates if row["original_selected"]
                ),
                "retrieved_adjacent_count": sum(
                    1 for row in candidates if row["adjacent_completion"]
                ),
                "selected_evidence_count_before_policy": len(candidates),
                "policy_primary_candidate": trace["primary_citation_ids"],
                "policy_supporting_candidates": trace["supporting_citation_ids"],
                "final_cited_relations": sorted(cited_keys),
                "core_gold_relations_offline_only": sorted(sets["core"]),
                "equivalent_valid_relations_offline_only": sorted(sets["equivalent"]),
                "core_relation_retrieved": has_any(registry_keys, sets["core"]),
                "core_relation_selected": has_any(selected_keys, sets["core"]),
                "core_relation_cited": has_any(cited_keys, sets["core"]),
                "equivalent_relation_retrieved": has_any(registry_keys, sets["equivalent"]),
                "equivalent_relation_selected": has_any(selected_keys, sets["equivalent"]),
                "equivalent_relation_cited": has_any(cited_keys, sets["equivalent"]),
                "any_valid_retrieved": retrieved_any,
                "any_valid_candidate": candidate_any,
                "any_valid_selected": selected_any,
                "any_valid_cited": cited_any,
                "obligation_coverage": trace.get("uncovered_requirements") == [],
                "numeric_completeness": trace["numeric_validation"]["complete"],
                "comparison_completeness": trace["comparison_validation"]["complete"],
                "failure_stage": stage,
                "primary_root_cause": primary,
                "secondary_causes": secondary,
                "generic_fix_category": stage,
                "retrieval_change_needed": stage == "retrieval",
                "candidate_allocation_change_needed": stage == "candidate_pruning",
                "citation_selection_change_needed": stage == "selection",
                "narrowing_change_needed": stage == "claim_fallback",
                "no_online_gold_dependency": True,
                "citation_cap_blocked": cap_blocked,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    counts = Counter(row["primary_root_cause"] for row in rows)
    stage_counts = Counter(row["failure_stage"] for row in rows)
    metrics = {
        "schema_version": "dev-v3-6-evidence-funnel-metrics-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "diagnostic_only": True,
        "gold_used_for_offline_scoring_only": True,
        "total_required_claims": total,
        "f2_retrieval_misses": sum(not row["any_valid_retrieved"] for row in rows),
        "f3_candidate_pruning_misses": sum(
            row["any_valid_retrieved"] and not row["any_valid_candidate"] for row in rows
        ),
        "f5_policy_selection_misses": sum(
            row["any_valid_candidate"] and not row["any_valid_selected"] for row in rows
        ),
        "f6_selected_not_cited": sum(
            row["any_valid_selected"] and not row["any_valid_cited"] for row in rows
        ),
        "f7_support_completeness_failures": sum(
            row["any_valid_cited"] and not row["core_relation_cited"] for row in rows
        ),
        "narrowing_losses": sum(
            row["narrowed"] and row["primary_root_cause"] == "unnecessary_narrowing"
            for row in rows
        ),
        "unsupported_losses": sum(
            row["final_status"] == "unsupported"
            and row["primary_root_cause"] == "unnecessary_unsupported"
            for row in rows
        ),
        "primary_root_cause_distribution": dict(sorted(counts.items())),
        "failure_stage_distribution": dict(sorted(stage_counts.items())),
        "retrieval_upper_bound_core_recall": sum(row["core_relation_retrieved"] for row in rows)
        / total,
        "candidate_set_upper_bound_core_recall": sum(row["core_relation_selected"] for row in rows)
        / total,
        "policy_input_upper_bound_recall": sum(row["any_valid_candidate"] for row in rows) / total,
        "selected_evidence_recall": sum(row["any_valid_selected"] for row in rows) / total,
        "final_citation_recall": sum(row["any_valid_cited"] for row in rows) / total,
        "any_valid_retrieval_upper_bound": sum(row["any_valid_retrieved"] for row in rows)
        / total,
        "any_valid_candidate_upper_bound": sum(row["any_valid_candidate"] for row in rows)
        / total,
        "any_valid_final_recall": sum(row["any_valid_cited"] for row in rows) / total,
        "claim_obligation_achievable_coverage": sum(row["obligation_coverage"] for row in rows)
        / total,
        "citation_budget_constrained_achievable_coverage": sum(
            not row["citation_cap_blocked"] for row in rows
        )
        / total,
        "can_selection_only_cross_0296296": (
            sum(row["any_valid_candidate"] for row in rows) / total
        )
        >= 0.296296,
        "retrieval_completion_v2_required": (
            sum(row["any_valid_candidate"] for row in rows) / total
        )
        < 0.296296,
        "primary_quality_bottleneck": "MIXED"
        if len(stage_counts) > 1
        else next(iter(stage_counts), "UNKNOWN").upper(),
        "replay_hash": "",
    }
    stable = {key: value for key, value in metrics.items() if key in {
        "total_required_claims",
        "f2_retrieval_misses",
        "f3_candidate_pruning_misses",
        "f5_policy_selection_misses",
        "f6_selected_not_cited",
        "f7_support_completeness_failures",
        "narrowing_losses",
        "unsupported_losses",
        "primary_root_cause_distribution",
        "failure_stage_distribution",
        "any_valid_candidate_upper_bound",
        "any_valid_final_recall",
    }}
    metrics["replay_hash"] = canonical_hash(stable)
    return metrics


def write_outputs(rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    OUT_JSONL.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    aggregate = {
        "schema_version": "dev-v3-6-evidence-funnel-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "records": len(rows),
        "rows_hash": canonical_hash(rows),
        "metrics_hash": metrics["replay_hash"],
        "diagnostic_only": True,
        "no_online_gold_dependency": True,
        "rows": rows,
    }
    OUT_JSON.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "question_id",
            "required_claim_id",
            "final_status",
            "narrowed",
            "any_valid_retrieved",
            "any_valid_candidate",
            "any_valid_selected",
            "any_valid_cited",
            "core_relation_cited",
            "primary_root_cause",
            "failure_stage",
            "citation_cap_blocked",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    METRICS_JSON.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_DOC.write_text(
        "# Dev v3.6 Evidence Funnel\n\n"
        f"- Required claims: `{len(rows)}`\n"
        f"- Rows hash: `{aggregate['rows_hash']}`\n"
        f"- Diagnostic only: `true`\n"
        "- Gold/equivalent relations are used only for offline scoring and are not available to "
        "online selection.\n\n"
        "## Primary root causes\n\n"
        + "\n".join(
            f"- `{key}`: {value}"
            for key, value in metrics["primary_root_cause_distribution"].items()
        )
        + "\n",
        encoding="utf-8",
    )
    METRICS_DOC.write_text(
        "# Dev v3.6 Evidence Funnel Metrics\n\n"
        f"- F2 retrieval misses: `{metrics['f2_retrieval_misses']}`\n"
        f"- F3 candidate pruning misses: `{metrics['f3_candidate_pruning_misses']}`\n"
        f"- F5 selection misses: `{metrics['f5_policy_selection_misses']}`\n"
        f"- F6 selected-not-cited: `{metrics['f6_selected_not_cited']}`\n"
        f"- F7 support completeness failures: `{metrics['f7_support_completeness_failures']}`\n"
        f"- Any-valid retrieval upper bound: `{metrics['any_valid_retrieval_upper_bound']}`\n"
        f"- Any-valid candidate upper bound: `{metrics['any_valid_candidate_upper_bound']}`\n"
        f"- Any-valid final recall: `{metrics['any_valid_final_recall']}`\n"
        f"- Selection-only can cross 0.296296: "
        f"`{metrics['can_selection_only_cross_0296296']}`\n"
        f"- Retrieval completion required: `{metrics['retrieval_completion_v2_required']}`\n"
        f"- Replay hash: `{metrics['replay_hash']}`\n\n"
        "Upper bounds are diagnostic and do not imply online availability. Gold is not injected "
        "into any production policy.\n",
        encoding="utf-8",
    )


def main() -> None:
    first_rows = build_rows()
    second_rows = build_rows()
    if canonical_hash(first_rows) != canonical_hash(second_rows):
        raise RuntimeError("DEV_V3_6_EVIDENCE_FUNNEL_NOT_DETERMINISTIC")
    if len(first_rows) != 27:
        raise RuntimeError("DEV_V3_6_EVIDENCE_FUNNEL_RECORD_COUNT")
    metrics = summarize(first_rows)
    write_outputs(first_rows, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
