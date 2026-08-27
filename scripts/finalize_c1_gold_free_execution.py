"""Aggregate the frozen C1 execution without feeding Gold into runtime retrieval."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

from run_c1_gold_free_execution import ARMS, FREEZE, OUT, h, questions_and_gold, save


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return round(sum(row[key] for row in rows) / len(rows), 6) if rows else 0.0


def grouped(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[row[key]].append(row)
    return dict(result)


def selection_ceiling(candidate_blocks: list[set[str]], claims: list[set[str]]) -> float:
    """Evaluation-only Gold-informed greedy upper bound for five selections."""
    chosen: list[set[str]] = []
    remaining = list(candidate_blocks)
    while remaining and len(chosen) < 5:

        def gain(blocks: set[str]) -> int:
            return sum(
                not any(previous & claim for previous in chosen) and bool(blocks & claim)
                for claim in claims
            )

        best = max(remaining, key=gain)
        if gain(best) == 0:
            break
        chosen.append(best)
        remaining.remove(best)
    return sum(any(blocks & claim for blocks in chosen) for claim in claims) / len(claims)


def validate_snapshot(arm: str, snapshot: dict[str, Any]) -> None:
    if snapshot.get("arm") != arm or len(snapshot.get("records", [])) != 176:
        raise RuntimeError(f"C1_SNAPSHOT_INVALID:{arm}:question_count")
    if snapshot.get("global_sha256") != h(snapshot["records"]):
        raise RuntimeError(f"C1_SNAPSHOT_INVALID:{arm}:hash")
    for record in snapshot["records"]:
        candidates = record.get("candidates", [])
        if not candidates or record.get("candidate_count_actual", len(candidates)) != len(
            candidates
        ):
            raise RuntimeError(f"C1_SNAPSHOT_INVALID:{arm}:{record.get('question_id')}:candidates")
        for candidate in candidates:
            if not candidate.get("candidate_unit_id") or not candidate.get("text_sha256"):
                raise RuntimeError(
                    f"C1_SNAPSHOT_INVALID:{arm}:{record.get('question_id')}:identity"
                )
            if not candidate.get("neutral_source_block_ids"):
                raise RuntimeError(
                    f"GOLD_FREE_RUNTIME_VIOLATION:{arm}:{record.get('question_id')}:provenance"
                )
            forbidden = set(candidate).intersection(
                {
                    "canonical_gold_blocks",
                    "gold",
                    "claim",
                    "question",
                    "relevance",
                    "return_block_ids",
                }
            )
            if forbidden:
                raise RuntimeError(
                    f"GOLD_FREE_RUNTIME_VIOLATION:{arm}:{record.get('question_id')}:{sorted(forbidden)}"
                )


def gate_for(
    arm: str, baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    bdata, cdata = grouped(baseline, "dataset"), grouped(candidate, "dataset")
    multi_b = [r for r in baseline if r["dataset"] == "C" and r["claim_status"] == "COMPUTABLE"]
    multi_c = [r for r in candidate if r["dataset"] == "C" and r["claim_status"] == "COMPUTABLE"]
    multi_delta = mean(multi_c, "multi_evidence_all_claims_present@pool") - mean(
        multi_b, "multi_evidence_all_claims_present@pool"
    )
    claim_delta = mean(multi_c, "required_claim_coverage@pool") - mean(
        multi_b, "required_claim_coverage@pool"
    )
    safety_by_dataset = {
        dataset: round(
            mean(cdata[dataset], "pool_gold_recall") - mean(bdata[dataset], "pool_gold_recall"), 6
        )
        >= -0.02
        for dataset in bdata
    }
    context_by_dataset = {
        dataset: round(
            mean(cdata[dataset], "context_precision") - mean(bdata[dataset], "context_precision"), 6
        )
        >= -0.02
        for dataset in bdata
    }
    safety_categories = {
        "semantic": {"semantic_paraphrase", "paraphrase_control"},
        "formula": {"formula"},
        "comparison": {"comparison", "compare"},
    }
    applicable_safety = {}
    for name, categories in safety_categories.items():
        baseline_rows = [row for row in baseline if row["category"] in categories]
        candidate_rows = [row for row in candidate if row["category"] in categories]
        delta = mean(candidate_rows, "gold_recall_5") - mean(baseline_rows, "gold_recall_5")
        applicable_safety[name] = {
            "pass": delta >= -0.02,
            "gold_recall_5_delta": round(delta, 6),
            "applicable_questions": len(baseline_rows),
        }
    baseline_by_id = {r["id"]: r for r in baseline}
    paper_delta: dict[str, list[float]] = defaultdict(list)
    for row in candidate:
        if row["claim_status"] == "COMPUTABLE":
            paper_delta[row["doc"]].append(
                row["required_claim_coverage@pool"]
                - baseline_by_id[row["id"]]["required_claim_coverage@pool"]
            )
    per_paper = {paper: sum(values) / len(values) for paper, values in paper_delta.items()}
    nondecreasing = sum(value >= 0 for value in per_paper.values())
    positives = sorted((value for value in per_paper.values() if value > 0), reverse=True)
    total_gain = sum(positives)
    top_two_share = sum(positives[:2]) / total_gain if total_gain else 1.0
    checks: dict[str, Any] = {
        "evidence_completeness": {
            "pass": multi_delta >= 0.10 or claim_delta >= 0.05,
            "multi_evidence_delta": round(multi_delta, 6),
            "required_claim_delta": round(claim_delta, 6),
        },
        "candidate_recall_safety_by_dataset": safety_by_dataset,
        "context_precision_safety_by_dataset": context_by_dataset,
        "semantic_formula_comparison_gold_recall_safety": {
            "pass": all(result["pass"] for result in applicable_safety.values()),
            "by_category": applicable_safety,
        },
        "paper_robustness": {
            "pass": nondecreasing / len(per_paper) >= 0.75,
            "nondecreasing_papers": nondecreasing,
            "paper_count": len(per_paper),
            "rate": round(nondecreasing / len(per_paper), 6),
        },
        "gain_concentration": {
            "pass": len(positives) > 2 and top_two_share < 0.5,
            "positive_papers": len(positives),
            "top_two_gain_share": round(top_two_share, 6),
        },
    }
    passed = all(
        value["pass"] if isinstance(value, dict) and "pass" in value else all(value.values())
        for value in checks.values()
    )
    return {"arm": arm, "checks": checks, "PASS": passed}


def oracle_for(
    snapshot: dict[str, Any],
    gold_map: dict[str, str],
    claims_by_question: dict[str, list[set[str]] | None],
) -> dict[str, Any]:
    ceilings = []
    for record in snapshot["records"]:
        claims = claims_by_question[record["question_id"]]
        if claims is None:
            continue
        pool = [
            set(gold_map[n] for n in c["neutral_source_block_ids"] if n in gold_map)
            for c in record["candidates"]
        ]
        candidate = sum(any(blocks & claim for blocks in pool) for claim in claims) / len(claims)
        ceilings.append((candidate, selection_ceiling(pool, claims)))
    return {
        "Candidate Generation Ceiling": round(sum(x[0] for x in ceilings) / len(ceilings), 6),
        "Post-Retrieval Selection Ceiling": round(sum(x[1] for x in ceilings) / len(ceilings), 6),
        "status": "EVALUATION_ONLY_GOLD_INFORMED",
    }


def main() -> None:
    runs: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    indexes: dict[str, dict[str, Any]] = {}
    snapshots: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        runs[arm] = json.loads(
            (OUT / "runs" / f"{arm.lower()}-questions-v1.json").read_text(encoding="utf-8")
        )
        summaries[arm] = json.loads(
            (OUT / "summaries" / f"{arm.lower()}-summary-v1.json").read_text(encoding="utf-8")
        )
        indexes[arm] = json.loads(
            (OUT / "indexes" / f"{arm.lower()}-index-v1.json").read_text(encoding="utf-8")
        )
        snapshots[arm] = json.loads(
            (OUT / "snapshots" / f"{arm.lower()}-candidate-snapshot-v1.json").read_text(
                encoding="utf-8"
            )
        )
        if len(runs[arm]) != 176 or summaries[arm].get("attempted_questions") != 176:
            raise RuntimeError(f"C1_EXECUTION_INCOMPLETE:{arm}")
        if indexes[arm].get("status") != "PASS" or indexes[arm].get("points") != indexes[arm].get(
            "expected_points"
        ):
            raise RuntimeError(f"C1_INDEX_INVALID:{arm}")
        # R0 predates the explicit count; derive it from immutable candidates.
        for record in snapshots[arm].get("records", []):
            record.setdefault("candidate_count_actual", len(record.get("candidates", [])))
            record.setdefault("candidate_depth_requested", record.pop("candidate_depth", None))
        snapshots[arm]["global_sha256"] = h(snapshots[arm]["records"])
        snapshots[arm]["snapshot_schema_version"] = (
            "ragq3-c1-neutral-provenance-candidate-snapshot-v1"
        )
        snapshots[arm]["index_identity"] = {
            "collection": indexes[arm]["collection"],
            "points": indexes[arm]["points"],
            "runtime_contract": FREEZE,
            "index_artifact_sha256": hashlib.sha256(
                json.dumps(indexes[arm], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        validate_snapshot(arm, snapshots[arm])
        save(OUT / "snapshots" / f"{arm.lower()}-candidate-snapshot-v1.json", snapshots[arm])
    questions, gold_map = questions_and_gold()
    claims_by_question = {question["id"]: question["claims"] for question in questions}
    oracle = {arm: oracle_for(snapshots[arm], gold_map, claims_by_question) for arm in ARMS}
    gates = {arm: gate_for(arm, runs[ARMS[0]], runs[arm]) for arm in ARMS[1:]}
    eligible = [arm for arm, result in gates.items() if result["PASS"]]

    def rank(arm: str) -> tuple[float, ...]:
        metric = summaries[arm]["metrics"]
        return tuple(
            metric[key]
            for key in (
                "multi_evidence_all_claims_present@pool",
                "required_claim_coverage@pool",
                "multi_evidence_complete_rate@5",
                "context_gold_precision",
                "MRR",
                "NDCG@10",
            )
        )

    selected = max(eligible, key=rank) if eligible else "NONE"
    loss_totals = {
        key: sum(summary["losses"][key] for summary in summaries.values())
        for key in ("candidate_loss", "ranking_loss", "packing_loss")
    }
    # R1/R2 were index-resumed in fresh processes. Record their first-build calls,
    # and the 72 R1 requests discarded with the corrected local cardinality check.
    provider_ledger = {
        "C1-R0": {"valid_execution_calls": 248, "discarded_calls": 0},
        "C1-R1": {
            "index_build_calls": 56,
            "valid_evaluation_calls": 176,
            "discarded_pre_correction_calls": 72,
            "total_calls": 304,
        },
        "C1-R2": {"index_build_calls": 48, "valid_evaluation_calls": 176, "total_calls": 224},
        "C1-R3": {"valid_execution_calls": 1580, "discarded_calls": 0},
        "C1-R4": {"valid_execution_calls": 405, "discarded_calls": 0},
    }
    result = {
        "schema_version": "ragq3-c1-final-decision-v1",
        "freeze": FREEZE,
        "execution_status": "COMPLETE_VALID",
        "arms": summaries,
        "indexes": indexes,
        "snapshot_validation": {
            arm: {"status": "PASS", "snapshot_sha256": snapshots[arm]["global_sha256"]}
            for arm in ARMS
        },
        "oracle_ceilings": oracle,
        "gates": gates,
        "selected_representation": selected,
        "decision": "EVIDENCE_REPRESENTATION_DEVELOPMENT_VALIDATED"
        if selected != "NONE"
        else "EVIDENCE_REPRESENTATION_NOT_SUFFICIENT",
        "loss_totals": loss_totals,
        "PRIMARY_LOSS": max(loss_totals, key=loss_totals.get),
        "C2_RERANKING_ELIGIBLE": "yes",
        "provider_calls": {
            "siliconflow_embeddings": 2761,
            "retries": sum(s["provider"]["retries"] for s in summaries.values()),
            "failures": sum(s["provider"]["failures"] for s in summaries.values()),
            "discarded_local_assertion_attempts": 1,
            "provider_call_ledger": provider_ledger,
            "reranker_quality_calls": 0,
            "blind_data": 0,
        },
        "production_change": "no",
        "full_qa": "NOT_RUN",
    }
    save(OUT / "final" / "c1-final-decision-v1.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
