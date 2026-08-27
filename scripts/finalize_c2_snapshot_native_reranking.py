"""Finalize the frozen C2 paired reranking decision from sealed-snapshot evidence."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from run_c1_gold_free_execution import h, questions_and_gold

ROOT = Path(__file__).resolve().parents[1]
C1 = ROOT / "artifacts" / "rag-quality-v3" / "c1" / "execution"
OUT = ROOT / "artifacts" / "rag-quality-v3" / "c2" / "execution"
FREEZE_DIR = ROOT / "artifacts" / "rag-quality-v3" / "c2" / "preregistration"
ARMS = ("C2-R0", "C2-R1")


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def mean(rows: list[dict], key: str) -> float:
    return round(sum(row[key] for row in rows) / len(rows), 6) if rows else 0.0


def groups(rows: list[dict], key: str) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        result[row[key]].append(row)
    return dict(result)


def greedy_gold_ceiling(pool: list[set[str]], gold: set[str]) -> float:
    selected: list[set[str]] = []
    remaining = list(pool)
    while remaining and len(selected) < 5:
        best = max(remaining, key=lambda item: len(item & gold - set().union(*selected)))
        if not best & gold - set().union(*selected):
            break
        selected.append(best)
        remaining.remove(best)
    return len(set().union(*selected) & gold) / len(gold)


def greedy_claim_ceiling(pool: list[set[str]], claims: list[set[str]]) -> float:
    selected: list[set[str]] = []
    remaining = list(pool)
    while remaining and len(selected) < 5:

        def gain(item: set[str]) -> int:
            return sum(
                not any(old & claim for old in selected) and bool(item & claim) for claim in claims
            )

        best = max(remaining, key=gain)
        if gain(best) == 0:
            break
        selected.append(best)
        remaining.remove(best)
    return sum(any(item & claim for item in selected) for claim in claims) / len(claims)


def verify_freeze() -> dict:
    freeze = json.loads((FREEZE_DIR / "c2-pre-result-freeze-v1.json").read_text(encoding="utf-8"))
    value = dict(freeze)
    expected = value.pop("freeze_sha256", None)
    actual = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if freeze.get("status") != "FROZEN_PRE_RESULT" or expected != actual:
        raise RuntimeError("C2_PRE_RESULT_FREEZE_INVALID")
    return freeze


def verify_pairs(traces: dict[str, dict], sealed: dict) -> dict:
    sealed_by_id = {record["question_id"]: record for record in sealed["records"]}
    pair_count = 0
    for question_id, original in sealed_by_id.items():
        original_ids = [candidate["candidate_unit_id"] for candidate in original["candidates"]]
        r0 = next(
            record for record in traces["C2-R0"]["records"] if record["question_id"] == question_id
        )
        r1 = next(
            record for record in traces["C2-R1"]["records"] if record["question_id"] == question_id
        )
        if r0["candidate_unit_ids"] != original_ids:
            raise RuntimeError(f"C2_PAIRED_IDENTITY_VIOLATION:R0:{question_id}")
        if set(r1["candidate_unit_ids"]) != set(original_ids) or len(
            r1["candidate_unit_ids"]
        ) != len(original_ids):
            raise RuntimeError(f"C2_PAIRED_IDENTITY_VIOLATION:R1:{question_id}")
        if r0["pool_sha256"] != r1["pool_sha256"]:
            raise RuntimeError(f"C2_PAIRED_IDENTITY_VIOLATION:pool:{question_id}")
        pair_count += 1
    return {"status": "PASS", "questions": pair_count, "candidate_membership": "IDENTICAL"}


def oracle(sealed: dict) -> dict:
    questions, gold_map = questions_and_gold()
    question_by_id = {question["id"]: question for question in questions}
    gold_values, claim_values, multi_values = [], [], []
    for record in sealed["records"]:
        question = question_by_id[record["question_id"]]
        pool = [
            {gold_map[item] for item in candidate["neutral_source_block_ids"] if item in gold_map}
            for candidate in record["candidates"]
        ]
        gold_values.append(greedy_gold_ceiling(pool, question["gold"]))
        if question["claims"] is not None:
            claim = greedy_claim_ceiling(pool, question["claims"])
            claim_values.append(claim)
            if question["category"] == "multi_evidence":
                multi_values.append(float(claim == 1.0))
    return {
        "GoldR@5": round(sum(gold_values) / len(gold_values), 6),
        "required_claim_coverage@5": round(sum(claim_values) / len(claim_values), 6),
        "multi_evidence_complete_rate@5": round(sum(multi_values) / len(multi_values), 6),
        "status": "EVALUATION_ONLY_GOLD_INFORMED_SEALED_POOL",
    }


def subset_safety(r0: list[dict], r1: list[dict]) -> dict:
    base = {row["id"]: row for row in r0}
    categories = (
        "multi_evidence",
        "semantic_paraphrase",
        "formula",
        "comparison",
        "single_evidence_control",
    )
    result = {}
    for category in categories:
        rows = [row for row in r1 if row["category"] == category]
        delta = mean(rows, "gold_recall_5") - mean(
            [base[row["id"]] for row in rows], "gold_recall_5"
        )
        result[category] = {
            "questions": len(rows),
            "gold_recall_5_delta": round(delta, 6),
            "pass": delta >= -0.02,
        }
    return result


def paper_robustness(r0: list[dict], r1: list[dict]) -> dict:
    base = {row["id"]: row for row in r0}
    deltas: dict[str, list[float]] = defaultdict(list)
    for row in r1:
        if row["claim_status"] == "COMPUTABLE":
            deltas[row["doc"]].append(
                row["required_claim_coverage@5"] - base[row["id"]]["required_claim_coverage@5"]
            )
    per_paper = {paper: sum(values) / len(values) for paper, values in deltas.items()}
    nondecreasing = sum(value >= 0 for value in per_paper.values())
    rate = nondecreasing / len(per_paper)
    return {
        "pass": rate >= 0.75,
        "nondecreasing_papers": nondecreasing,
        "paper_count": len(per_paper),
        "rate": round(rate, 6),
        "per_paper_delta": {key: round(value, 6) for key, value in per_paper.items()},
    }


def main() -> None:
    freeze = verify_freeze()
    summaries = {
        arm: json.loads(
            (OUT / "summaries" / f"{arm.lower()}-summary-v1.json").read_text(encoding="utf-8")
        )
        for arm in ARMS
    }
    runs = {
        arm: json.loads(
            (OUT / "runs" / f"{arm.lower()}-questions-v1.json").read_text(encoding="utf-8")
        )
        for arm in ARMS
    }
    traces = {
        arm: json.loads(
            (OUT / "traces" / f"{arm.lower()}-ordering-v1.json").read_text(encoding="utf-8")
        )
        for arm in ARMS
    }
    if any(
        summary.get("attempted_questions") != 176 or len(runs[arm]) != 176
        for arm, summary in summaries.items()
    ):
        raise RuntimeError("C2_EXECUTION_INCOMPLETE")
    if any(trace.get("global_sha256") != h(trace["records"]) for trace in traces.values()):
        raise RuntimeError("C2_TRACE_HASH_INVALID")
    sealed = json.loads(
        (C1 / "snapshots" / "c1-r0-candidate-snapshot-v1.json").read_text(encoding="utf-8")
    )
    paired = verify_pairs(traces, sealed)
    r0, r1 = runs["C2-R0"], runs["C2-R1"]
    m0, m1 = summaries["C2-R0"]["metrics"], summaries["C2-R1"]["metrics"]
    metric_keys = (
        "GoldR@5",
        "MRR",
        "NDCG@10",
        "context_precision",
        "context_recall",
        "required_claim_coverage@5",
        "multi_evidence_complete_rate@5",
    )
    deltas = {key: round(m1[key] - m0[key], 6) for key in metric_keys}
    subsets = subset_safety(r0, r1)
    robustness = paper_robustness(r0, r1)
    gates = {
        "overall_ranking": {
            "pass": deltas["MRR"] >= 0.02 and deltas["NDCG@10"] >= 0.02,
            "MRR_delta": deltas["MRR"],
            "NDCG@10_delta": deltas["NDCG@10"],
        },
        "gold_recall_safety": {"pass": deltas["GoldR@5"] >= -0.02, "delta": deltas["GoldR@5"]},
        "claim_safety": {
            "pass": deltas["required_claim_coverage@5"] >= -0.02,
            "delta": deltas["required_claim_coverage@5"],
        },
        "multi_evidence_safety": {
            "pass": deltas["multi_evidence_complete_rate@5"] >= -0.02,
            "delta": deltas["multi_evidence_complete_rate@5"],
        },
        "subset_safety": {
            "pass": all(value["pass"] for value in subsets.values()),
            "subsets": subsets,
        },
        "paper_robustness": robustness,
    }
    passed = all(value["pass"] for value in gates.values())
    ceilings = oracle(sealed)
    gaps = {
        key: round(ceilings[key] - m1[key], 6)
        for key in ("GoldR@5", "required_claim_coverage@5", "multi_evidence_complete_rate@5")
    }
    smoke = json.loads(
        (
            ROOT
            / "artifacts"
            / "rag-quality-v3"
            / "c2"
            / "preflight"
            / "siliconflow-reranker-smoke-v1.json"
        ).read_text(encoding="utf-8")
    )
    provider_calls = smoke["api_request_count"] + summaries["C2-R1"]["provider"]["logical_requests"]
    result = {
        "schema_version": "ragq3-c2-final-decision-v1",
        "freeze_commit": "e501e0b42bea4a159c98c7d84189c03bfaabf379",
        "freeze_sha256": freeze["freeze_sha256"],
        "execution_status": "COMPLETE_VALID",
        "paired_identity": paired,
        "metrics": {"C2-R0": m0, "C2-R1": m1, "delta": deltas},
        "losses": {
            "C2-R0": summaries["C2-R0"]["losses"],
            "C2-R1": summaries["C2-R1"]["losses"],
            "ranking_loss_recovered": summaries["C2-R0"]["losses"]["ranking_loss"]
            - summaries["C2-R1"]["losses"]["ranking_loss"],
        },
        "oracle_selection_ceiling": ceilings,
        "actual_to_oracle_gap": gaps,
        "subset_safety": subsets,
        "paper_robustness": robustness,
        "frozen_gate": {"PASS": passed, "checks": gates},
        "selected_candidate": "C2-R1" if passed else "NONE",
        "decision": "CLEAN_RERANKING_DEVELOPMENT_VALIDATED"
        if passed
        else "CLEAN_RERANKING_INTERVENTION_NOT_SUFFICIENT",
        "B1_BLIND_VALIDATION_ELIGIBLE": "yes" if passed else "no",
        "provider": {
            "smoke_http": smoke["http_status"],
            "calls": provider_calls,
            "failures": summaries["C2-R1"]["provider"]["failures"],
            "latency_ms": round(
                smoke["latency_ms"] + summaries["C2-R1"]["provider"]["latency_ms"], 3
            ),
        },
        "invariants": {
            "retrieval_calls": 0,
            "embedding_calls": 0,
            "blind_data": 0,
            "full_qa": "NOT_RUN",
            "production_change": "no",
        },
    }
    save(OUT / "final" / "c2-final-decision-v1.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
