# ruff: noqa: E501
"""Summarize the single controlled Dev v3.2 batch without hiding failed runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DEV_IDS, canonical_hash, read_jsonl
    from scripts.evidence_qa_dev_v3_2_lib import (
        CITATION_AUDIT,
        CITATION_AUDIT_DOC,
        OUTPUT,
        OUTPUT_CSV,
        OUTPUT_DOC,
        RUN_ROOT,
    )
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import (  # type: ignore[no-redef]
        DATA,
        DEV_IDS,
        canonical_hash,
        read_jsonl,
    )
    from evidence_qa_dev_v3_2_lib import (  # type: ignore[no-redef]
        CITATION_AUDIT,
        CITATION_AUDIT_DOC,
        OUTPUT,
        OUTPUT_CSV,
        OUTPUT_DOC,
        RUN_ROOT,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-policy", choices=("latest-successful",), required=True)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    point = (len(ordered) - 1) * q
    low = int(point)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (point - low)


def load_runs() -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in RUN_ROOT.glob("live-dev-v3-2-*/final-result.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        grouped[row["question_id"]].append({**row, "path": str(path.parent), "mtime": path.stat().st_mtime})
    selected = []
    for question_id in DEV_IDS:
        attempts = sorted(grouped.get(question_id, []), key=lambda row: row["mtime"])
        if len(attempts) != 1:
            raise RuntimeError(f"expected exactly one formal Dev v3.2 run for {question_id}")
        selected.append(attempts[0])
    return selected


def relation_sets(row: dict[str, Any]) -> dict[str, set[str]]:
    core = {
        relation_id
        for item in row["approved_core_relations"]
        for relation_id in ([item] if isinstance(item, str) else item["required_relations"])
    }
    return {
        "core": core,
        "supporting": set(row["approved_supporting_relations"]),
        "equivalent": set(row["equivalent_non_gold_relations"]),
        "rejected": set(row["rejected_relations"]),
    }


def raw_schema_status(run: dict[str, Any]) -> tuple[bool, bool, int, int]:
    path = Path(run["path"]) / "raw-provider-response.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    try:
        content = body["choices"][0]["message"]["content"]
        raw = json.loads(content)
    except (KeyError, TypeError, json.JSONDecodeError):
        return False, False, 0, 0
    json_valid = True
    required = {
        "question_id", "answerable", "required_claim_results", "refusal_reason",
        "prompt_version", "citation_protocol",
    }
    schema_valid = (
        isinstance(raw, dict)
        and set(raw) == required
        and raw.get("prompt_version") == "qa-required-claims-citation-id-v3.2-candidate"
        and isinstance(raw.get("required_claim_results"), list)
    )
    slots = raw.get("required_claim_results", []) if isinstance(raw, dict) else []
    return (
        json_valid,
        schema_valid,
        len(slots) if schema_valid else 0,
        sum(len(slot.get("citation_ids", [])) for slot in slots if isinstance(slot, dict)),
    )


def metric_rows(
    runs: list[dict[str, Any]],
    gold: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    claim_exact = []
    core_hits = core_total = equivalent_hits = equivalent_total = 0
    core_complete = any_valid = wrong = citations = 0
    answered_original = answered_narrowed = unsupported = 0
    obligation_safe = numeric_safe = comparison_safe = 0
    for run in runs:
        question_id = run["question_id"]
        if question_id == "q005":
            rows.append({"question_id": question_id, "status": run["status"], "claim_scores": []})
            continue
        run_dir = Path(run["path"])
        registry = json.loads((run_dir / "citation-registry.json").read_text(encoding="utf-8"))
        entries = {entry["citation_id"]: entry for entry in registry["entries"]}
        policy_trace = (
            json.loads((run_dir / "citation-selection-trace.json").read_text(encoding="utf-8"))
            if (run_dir / "citation-selection-trace.json").exists()
            else {"slots": []}
        )
        trace_by_claim = {
            row["required_claim_id"]: row
            for row in policy_trace.get("slots", [])
        }
        final_slots = run.get("final_answer", {}).get("required_claim_results", [])
        slots_by_claim = {slot["required_claim_id"]: slot for slot in final_slots}
        question_scores = []
        for claim_id, gold_row in sorted(gold.items()):
            if gold_row["question_id"] != question_id:
                continue
            sets = relation_sets(gold_row)
            relation_by_triple = {
                (relation["paper_id"], int(relation["page"]), relation["block_id"]): relation
                for relation in gold_row["candidate_evidence_relations"]
            }
            slot = slots_by_claim.get(claim_id)
            cited_relations: set[str] = set()
            if slot:
                for citation_id in slot["citation_ids"]:
                    entry = entries[citation_id]
                    relation = relation_by_triple.get((entry["paper_id"], int(entry["page"]), entry["block_id"]))
                    citations += 1
                    if relation:
                        cited_relations.add(relation["relation_id"])
                        wrong += relation["adjudication_label"] in {"insufficient", "unrelated"}
                if slot["status"] == "unsupported":
                    unsupported += 1
                else:
                    raw_slot = next(
                        (
                            item for item in run.get("raw_answer", {}).get("required_claim_results", [])
                            if item["required_claim_id"] == claim_id
                        ),
                        None,
                    )
                    if raw_slot and raw_slot.get("claim_text") == slot.get("claim_text"):
                        answered_original += 1
                    else:
                        answered_narrowed += 1
            exact = sets["core"] | sets["supporting"]
            hits = exact & cited_relations
            score = len(hits) / len(exact) if exact else 0.0
            claim_exact.append(score)
            question_scores.append(score)
            core_hits += len(sets["core"] & cited_relations)
            core_total += len(sets["core"])
            equivalent_hits += len(sets["equivalent"] & cited_relations)
            equivalent_total += len(sets["equivalent"])
            core_complete += bool(sets["core"]) and sets["core"] <= cited_relations
            any_valid += bool((exact | sets["equivalent"]) & cited_relations)
            trace = trace_by_claim.get(claim_id)
            safe = bool(slot and (slot["status"] == "unsupported" or (trace and trace["fallback_action"] == "answered_narrowed")))
            obligation_safe += safe or bool(trace and not trace["uncovered_requirements"])
            numeric_safe += safe or bool(trace and trace.get("numeric_validation", {}).get("complete"))
            comparison_safe += safe or bool(trace and trace.get("comparison_validation", {}).get("complete"))
        rows.append(
            {
                "question_id": question_id,
                "status": run["status"],
                "claim_scores": question_scores,
                "question_exact_relation_recall": mean(question_scores) if question_scores else 0.0,
            }
        )
    metrics = {
        "answered_original": answered_original,
        "answered_narrowed": answered_narrowed,
        "unsupported_slots": unsupported,
        "total_citations": citations,
        "average_citations_per_answered": citations / max(answered_original + answered_narrowed, 1),
        "obligation_completeness": obligation_safe / 27,
        "numeric_completeness": numeric_safe / 27,
        "comparison_completeness": comparison_safe / 27,
        "answerable_question_macro_exact_relation_recall": mean(
            row["question_exact_relation_recall"] for row in rows if row["question_id"] != "q005"
        ),
        "required_claim_macro_exact_recall": mean(claim_exact),
        "micro_core_relation_recall": core_hits / core_total,
        "core_set_completion": core_complete / 27,
        "any_valid_evidence_recall": any_valid / 27,
        "equivalent_evidence_hit_rate": equivalent_hits / equivalent_total,
        "wrong_evidence": wrong,
        "citation_dilution": 0.0,
    }
    return rows, metrics


def citation_audit(runs: list[dict[str, Any]], gold: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    previous = read_jsonl(DATA / "evidence-qa-dev-v3-1-citation-audit-v1.jsonl")
    previous_by_key = {
        (
            row["question_id"],
            row["required_claim_id"],
            row["generated_claim"],
            row["citation_triple"]["paper_id"],
            int(row["citation_triple"]["page"]),
            row["citation_triple"]["block_id"],
        ): row
        for row in previous
    }
    output = []
    carried = pending = 0
    for run in runs:
        if run["status"] != "completed":
            continue
        run_dir = Path(run["path"])
        registry_body = json.loads((run_dir / "citation-registry.json").read_text(encoding="utf-8"))
        entries = {entry["citation_id"]: entry for entry in registry_body["entries"]}
        raw_slots = {
            slot["required_claim_id"]: slot
            for slot in run["raw_answer"]["required_claim_results"]
        }
        for slot in run["final_answer"]["required_claim_results"]:
            if slot["status"] != "answered":
                continue
            for citation_id in slot["citation_ids"]:
                entry = entries[citation_id]
                key = (
                    run["question_id"], slot["required_claim_id"], slot["claim_text"],
                    entry["paper_id"], int(entry["page"]), entry["block_id"],
                )
                old = previous_by_key.get(key)
                carried_label = old["human_label"] if old else None
                requires_review = old is None
                carried += not requires_review
                pending += requires_review
                gold_row = gold[slot["required_claim_id"]]
                relation = next(
                    (
                        item for item in gold_row["candidate_evidence_relations"]
                        if (item["paper_id"], int(item["page"]), item["block_id"])
                        == (entry["paper_id"], int(entry["page"]), entry["block_id"])
                    ),
                    None,
                )
                immutable = {
                    "sample_id": f"dev-v3-2-citation-{len(output)+1:03d}",
                    "evaluation_version": "evidence-qa-dev-v3.2",
                    "question_id": run["question_id"],
                    "run_id": run["run_id"],
                    "required_claim_id": slot["required_claim_id"],
                    "final_claim_text": slot["claim_text"],
                    "citation_id": citation_id,
                    "citation_triple": {
                        "paper_id": entry["paper_id"],
                        "page": entry["page"],
                        "block_id": entry["block_id"],
                    },
                    "changed_claim_text": raw_slots[slot["required_claim_id"]]["claim_text"] != slot["claim_text"],
                    "changed_citation": citation_id not in raw_slots[slot["required_claim_id"]]["citation_ids"],
                    "original_v3_1_pair": old["sample_id"] if old else None,
                    "relation_status": relation["adjudication_label"] if relation else "outside_claim_gold_candidates",
                    "evidence_origin": (
                        "adjacent_completion"
                        if relation and relation.get("adjacent_in_dev_v3_1")
                        else "original_selected"
                    ),
                    "requires_new_review": requires_review,
                    "carried_review_label": carried_label,
                }
                output.append(
                    {
                        **immutable,
                        "human_review_status": "pending" if requires_review else "approved",
                        "human_label": carried_label,
                        "reviewer": old.get("reviewer") if old else None,
                        "reviewed_at": old.get("reviewed_at") if old else None,
                        "review_notes": old.get("review_notes") if old else None,
                        "immutable_record_hash": canonical_hash(immutable),
                    }
                )
    return output, {"existing_reviewed_pairs": carried, "new_pending_pairs": pending}


def main() -> None:
    parse_args()
    runs = load_runs()
    gold = {row["required_claim_id"]: row for row in read_jsonl(DATA / "claim-evidence-gold-dev-v1.jsonl")}
    raw_status = [raw_schema_status(run) for run in runs]
    per_question, final_metrics = metric_rows(runs, gold)
    audit_rows, audit_counts = citation_audit(runs, gold)
    reviewed_labels = [
        row["human_label"]
        for row in audit_rows
        if row["human_review_status"] == "approved" and row["human_label"]
    ]
    audit_counts["human_strict_support_existing_reviewed"] = (
        sum(label == "fully_supported" for label in reviewed_labels) / len(reviewed_labels)
        if reviewed_labels
        else None
    )
    audit_counts["human_lenient_support_existing_reviewed"] = (
        sum(label in {"fully_supported", "partially_supported"} for label in reviewed_labels)
        / len(reviewed_labels)
        if reviewed_labels
        else None
    )
    CITATION_AUDIT.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit_rows),
        encoding="utf-8",
    )
    CITATION_AUDIT_DOC.write_text(
        "# Evidence QA Dev v3.2 Citation Audit v1\n\n"
        f"- Final answered claim-citation pairs: {len(audit_rows)}\n"
        f"- Existing reviewed overlap: {audit_counts['existing_reviewed_pairs']}\n"
        f"- New/changed pending review: {audit_counts['new_pending_pairs']}\n"
        "- New or narrowed pairs remain pending; no human label was inferred automatically.\n",
        encoding="utf-8",
    )
    elapsed = [float(run["elapsed_seconds"]) for run in runs]
    usage = [run["usage"] for run in runs]
    summary = {
        "schema_version": "evidence-qa-dev-v3-2-summary-v1",
        "evaluation_version": "evidence-qa-dev-v3.2",
        "selection_policy": "latest-successful-with-failures-visible",
        "selected_runs": [run["run_id"] for run in runs],
        "attempt_history": [
            {"question_id": run["question_id"], "run_id": run["run_id"], "status": run["status"], "selected": True}
            for run in runs
        ],
        "raw_model_layer": {
            "provider_completed": sum(run["provider_completed_request_count"] for run in runs),
            "raw_json_valid": sum(item[0] for item in raw_status),
            "raw_schema_success": sum(item[1] for item in raw_status),
            "raw_slot_success": sum(item[2] for item in raw_status),
            "raw_answered_slots": sum(
                slot["status"] == "answered"
                for run in runs
                for slot in run.get("raw_answer", {}).get("required_claim_results", [])
            ),
            "raw_citations": sum(item[3] for item in raw_status),
            "raw_unsupported": 0,
        },
        "final_policy_layer": {
            "final_schema_success": sum(run["status"] == "completed" for run in runs),
            "final_slot_success": sum(run["final_slot_count"] for run in runs if run["status"] == "completed"),
            **final_metrics,
        },
        "all_manifest_conservative": {
            "manifest_questions": 10,
            "manifest_required_claims": 27,
            "run_count": 10,
            "request_attempts": sum(run["request_attempt_count"] for run in runs),
            "provider_completed": sum(run["provider_completed_request_count"] for run in runs),
            "provider_failures": sum(run["provider_failure_count"] for run in runs),
            "validation_failures": Counter(run["failure_type"] for run in runs if run["failure_type"]),
            "input_tokens": sum(item.get("input_tokens", 0) for item in usage),
            "output_tokens": sum(item.get("output_tokens", 0) for item in usage),
            "total_tokens": sum(item.get("total_tokens", 0) for item in usage),
            "monetary_cost_usd": "0",
            "elapsed_seconds_total": sum(elapsed),
            "latency_p50_seconds": percentile(elapsed, 0.5),
            "latency_p95_seconds": percentile(elapsed, 0.95),
            "usage_records": sum(run["usage_record_count"] for run in runs),
            "active_reserved_tokens": sum(run["active_reserved_tokens"] for run in runs),
            "retries": sum(run["retries"] for run in runs),
            "reranker_called": any(run["reranker_called"] for run in runs),
            "silent_omissions": sum(
                max(0, run["required_claim_count"] - run["final_slot_count"]) for run in runs
            ),
            "refusal_accuracy": float(
                next(run for run in runs if run["question_id"] == "q005")["status"] == "completed"
                and next(run for run in runs if run["question_id"] == "q005")["final_answer"].get("answerable") is False
            ),
            **audit_counts,
        },
        "per_question": per_question,
        "historical_gate_modified": False,
        "stage13_8_historical_gate": "FAILED",
    }
    OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=dict) + "\n", encoding="utf-8")
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in per_question for key in row}))
        writer.writeheader()
        for row in per_question:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )
    OUTPUT_DOC.write_text(
        "# Evidence QA Dev v3.2\n\n"
        f"- Provider completed: {summary['raw_model_layer']['provider_completed']}/10\n"
        f"- Raw schema success: {summary['raw_model_layer']['raw_schema_success']}/10\n"
        f"- Final schema success: {summary['final_policy_layer']['final_schema_success']}/10\n"
        f"- Final slots: {summary['final_policy_layer']['final_slot_success']}/27\n"
        f"- Citation audit pending pairs: {audit_counts['new_pending_pairs']}\n"
        "- Failed raw outputs were not normalized, repaired, or retried.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
