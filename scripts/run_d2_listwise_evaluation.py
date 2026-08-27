"""Frozen D2 listwise evidence-set selection over sealed C1/C2/B1 artifacts."""
# ruff: noqa
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from paper_research.config import Settings
from paper_research.providers.factory import build_llm_provider
from paper_research.providers.llm import LLMProviderError
from run_c1_gold_free_execution import questions_and_gold
from run_c2_snapshot_native_reranking import metrics, score_question

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = ROOT.parents[2]
OUT = ROOT / "artifacts/rag-quality-v3/d2/execution"
CHECKPOINT = OUT / "selector-checkpoint-v1.jsonl"
FINAL = OUT / "d2-final-decision-v1.json"
ARMS = ("D2-R0", "D2-R1")
CANDIDATE_DEPTH = 20
SELECTED_COUNT = 5
SYSTEM_PROMPT = """You are a neutral evidence-set selector. Return only one JSON object with exactly this schema:
{"selected_candidate_ids":["candidate-id-1","candidate-id-2","candidate-id-3","candidate-id-4","candidate-id-5"]}

Select exactly five unique candidate IDs from the supplied slate. Select the evidence set that most jointly supports a complete answer to the question. Consider direct relevance, complementary evidence, distinct required aspects, avoiding redundancy, and preserving useful cross-section evidence. Do not generate evidence, rewrite candidate text, infer unavailable facts, or include any ID not supplied. Do not return explanations or extra fields."""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def c1_cohort():
    snapshot = load_json(ROOT / "artifacts/rag-quality-v3/c1/execution/snapshots/c1-r0-candidate-snapshot-v1.json")
    trace = load_json(ROOT / "artifacts/rag-quality-v3/c2/execution/traces/c2-r1-ordering-v1.json")
    questions, gold_map = questions_and_gold()
    return "DEV176", snapshot, trace, {question["id"]: question for question in questions}, gold_map


def b1_cohort():
    snapshot = load_json(ROOT / "artifacts/rag-quality-v3/b1/execution/snapshots/b1-candidate-snapshot-v1.json")
    trace = load_json(ROOT / "artifacts/rag-quality-v3/b1/execution/traces/b1-r1-ordering-v1.json")
    runtime_questions = load_json(ROOT / "artifacts/rag-quality-v3/b1/pre-freeze/b1-blind-runtime-questions-v1.json")["questions"]
    gold = load_json(ROOT / "artifacts/rag-quality-v3/b1/pre-freeze/b1-blind-evaluation-gold-v1.json")["gold"]
    gold_by_question = {item["id"]: item for item in gold}
    questions = {
        item["id"]: {
            **item,
            "dataset": "DEVELOPMENT_VISIBLE_POST_BLIND_EVIDENCE",
            "gold": set(gold_by_question[item["id"]]["neutral_gold_unit_ids"]),
            "claims": [set(claim) for claim in gold_by_question[item["id"]]["required_claims"]],
        }
        for item in runtime_questions
    }
    gold_map = {unit: unit for item in gold for unit in item["neutral_gold_unit_ids"]}
    return "POSTBLIND60", snapshot, trace, questions, gold_map


def candidate_payload(candidate: dict[str, Any], rank: int, score: float | None) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_unit_id"],
        "candidate_text": candidate["text"],
        "canonical_document_id": candidate["canonical_document_id"],
        "source_spans": candidate["source_spans"],
        "reranker_rank": rank,
        "reranker_score": score,
    }


def selector_user_prompt(question: dict[str, Any], ordered: list[dict[str, Any]], scores: dict[str, float]) -> str:
    slate = [
        candidate_payload(candidate, rank, scores.get(candidate["candidate_unit_id"]))
        for rank, candidate in enumerate(ordered, start=1)
    ]
    return json.dumps({"question": question["query"], "candidates": slate}, ensure_ascii=False)


def load_checkpoint() -> dict[str, dict[str, Any]]:
    if not CHECKPOINT.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        records[item["key"]] = item
    for record in records.values():
        if (
            record.get("status") == "HARD_PROVIDER_ERROR"
            and record.get("error_code") == "STRUCTURED_JSON_RESPONSE_ERROR"
        ):
            record["status"] = "FAIL_CLOSED_INVALID_SELECTOR_OUTPUT"
    return records


def append_checkpoint(record: dict[str, Any]) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def validate_selection(payload: dict[str, Any], allowed_ids: list[str]) -> list[str]:
    if set(payload) != {"selected_candidate_ids"}:
        raise ValueError("unexpected_schema_fields")
    selected = payload["selected_candidate_ids"]
    if not isinstance(selected, list) or len(selected) != SELECTED_COUNT:
        raise ValueError("wrong_selected_count")
    if not all(isinstance(item, str) for item in selected):
        raise ValueError("non_string_candidate_id")
    if len(set(selected)) != SELECTED_COUNT:
        raise ValueError("duplicate_candidate_id")
    if any(item not in allowed_ids for item in selected):
        raise ValueError("unknown_candidate_id")
    return selected


def select_or_fail_closed(
    *,
    provider: Any,
    cohort: str,
    question: dict[str, Any],
    ordered: list[dict[str, Any]],
    scores: dict[str, float],
    checkpoint: dict[str, dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    key = f"{cohort}:{question['id']}"
    allowed_ids = [candidate["candidate_unit_id"] for candidate in ordered]
    input_sha = sha({"question": question["query"], "ids": allowed_ids, "scores": scores})
    existing = checkpoint.get(key)
    if existing and existing.get("input_sha256") == input_sha:
        return list(existing["selected_candidate_ids"]), existing
    try:
        response = provider.generate_structured_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=selector_user_prompt(question, ordered, scores),
            schema_name="ragq3-d2-listwise-selector-v1",
            request_context={"task_id": f"d2-{cohort}-{question['id']}", "run_id": "ragq3-d2"},
            max_output_tokens=128,
        )
        selected = validate_selection(response.payload, allowed_ids)
        record = {
            "key": key,
            "cohort": cohort,
            "question_id": question["id"],
            "input_sha256": input_sha,
            "selected_candidate_ids": selected,
            "status": "SUCCESS",
            "provider": response.provider,
            "model": response.model,
            "request_attempt_count": response.request_attempt_count,
            "retry_count": response.retry_count,
            "latency_ms": response.total_latency_ms,
            "usage": response.usage.model_dump(),
            "response_sha256": sha(response.payload),
        }
    except LLMProviderError as exc:
        # Truncated/invalid structured output is explicitly fail-closed; network
        # and provider transport errors remain hard blockers under the freeze.
        if exc.error_code == "STRUCTURED_JSON_RESPONSE_ERROR":
            record = {
                "key": key,
                "cohort": cohort,
                "question_id": question["id"],
                "input_sha256": input_sha,
                "selected_candidate_ids": allowed_ids[:SELECTED_COUNT],
                "status": "FAIL_CLOSED_INVALID_SELECTOR_OUTPUT",
                "failure_reason": exc.error_code,
                "request_attempt_count": exc.api_request_count,
                "retry_reasons": exc.retry_reasons,
            }
            append_checkpoint(record)
            return list(record["selected_candidate_ids"]), record
        record = {
            "key": key,
            "cohort": cohort,
            "question_id": question["id"],
            "input_sha256": input_sha,
            "selected_candidate_ids": allowed_ids[:SELECTED_COUNT],
            "status": "HARD_PROVIDER_ERROR",
            "error_code": exc.error_code,
            "stage": exc.stage,
            "error": str(exc)[:1000],
            "request_attempt_count": exc.api_request_count,
            "retry_reasons": exc.retry_reasons,
        }
        append_checkpoint(record)
        raise RuntimeError(f"D2_LLM_PROVIDER_BLOCKED:{key}:{exc.error_code}") from exc
    except (TypeError, ValueError) as exc:
        record = {
            "key": key,
            "cohort": cohort,
            "question_id": question["id"],
            "input_sha256": input_sha,
            "selected_candidate_ids": allowed_ids[:SELECTED_COUNT],
            "status": "FAIL_CLOSED_INVALID_SELECTOR_OUTPUT",
            "failure_reason": str(exc),
            "request_attempt_count": 1,
        }
    append_checkpoint(record)
    return list(record["selected_candidate_ids"]), record


def ordered_with_selected_first(ordered: list[dict[str, Any]], selected_ids: list[str]) -> list[dict[str, Any]]:
    by_id = {candidate["candidate_unit_id"]: candidate for candidate in ordered}
    selected = [by_id[item] for item in selected_ids]
    chosen = set(selected_ids)
    return [*selected, *(candidate for candidate in ordered if candidate["candidate_unit_id"] not in chosen)]


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return round(sum(item[key] for item in rows) / len(rows), 6)


def safety(base: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    base_by_id = {item["id"]: item for item in base}
    subsets: dict[str, float] = {}
    predicates = {
        "single_evidence": lambda item: item["category"].startswith("single"),
        "multi_evidence": lambda item: item["category"] == "multi_evidence",
        "semantic": lambda item: "paraphrase" in item["category"],
        "formula": lambda item: item["category"] == "formula",
        "comparison": lambda item: item["category"] in {"comparison", "compare"},
    }
    for name, predicate in predicates.items():
        rows = [item for item in candidate if predicate(item)]
        delta = mean(rows, "gold_recall_5") - mean([base_by_id[item["id"]] for item in rows], "gold_recall_5") if rows else 0.0
        subsets[name] = round(delta, 6)
    papers: dict[str, list[float]] = defaultdict(list)
    for item in candidate:
        if item["required_claim_coverage@5"] is not None:
            papers[item["doc"]].append(item["required_claim_coverage@5"] - base_by_id[item["id"]]["required_claim_coverage@5"])
    paper_rate = sum(sum(values) / len(values) >= 0 for values in papers.values()) / len(papers)
    tail = sum(item["gold_recall_5"] < base_by_id[item["id"]]["gold_recall_5"] for item in candidate)
    return {
        "subsets": subsets,
        "paper_robustness_rate": round(paper_rate, 6),
        "new_goldr_tail_regressions": tail,
        "pass": all(value >= -0.02 for value in subsets.values()) and paper_rate >= 0.75 and tail <= 2,
    }


def residual_recovery(rows_by_arm: dict[str, list[dict[str, Any]]], b1_snapshot: dict[str, Any], b1_trace: dict[str, Any]) -> dict[str, dict[str, dict[str, int]]]:
    residuals = load_json(ROOT / "artifacts/rag-quality-v3/d0/d0-post-blind-failure-attribution-v1.json")["residual_details"]
    orders = {item["question_id"]: item["candidate_unit_ids"] for item in b1_trace["records"]}
    snapshots = {item["question_id"]: item for item in b1_snapshot["records"]}
    rows = {item["id"]: item for item in rows_by_arm["D2-R1"]}
    output: dict[str, dict[str, int]] = defaultdict(lambda: {"fixed": 0, "unchanged": 0, "newly_regressed": 0})
    for residual in residuals:
        question_id = residual["question_id"]
        candidate_ids = set(rows[question_id]["selected_candidate_ids"])
        snapshot = snapshots[question_id]
        by_id = {candidate["candidate_unit_id"]: candidate for candidate in snapshot["candidates"]}
        baseline = [by_id[item] for item in orders[question_id]][:5]
        required = set(residual["missing_claim"])
        baseline_hit = any(set(item["neutral_source_block_ids"]) & required for item in baseline)
        selected_hit = any(set(by_id[item]["neutral_source_block_ids"]) & required for item in candidate_ids)
        bucket = output[residual["taxonomy"]]
        bucket["fixed" if selected_hit and not baseline_hit else "newly_regressed" if baseline_hit and not selected_hit else "unchanged"] += 1
    return dict(output)


def oracle_gaps(
    cohort_rows: dict[str, dict[str, list[dict[str, Any]]]],
    all_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    c2 = load_json(ROOT / "artifacts/rag-quality-v3/c2/execution/final/c2-final-decision-v1.json")
    b1 = load_json(ROOT / "artifacts/rag-quality-v3/b1/execution/final/b1-final-decision-v1.json")
    ceilings = {
        "DEV176": c2["oracle_selection_ceiling"],
        "POSTBLIND60": {
            "GoldR@5": b1["oracle_ceilings"]["GoldR@5"],
            "required_claim_coverage@5": b1["oracle_ceilings"]["Claim@5"],
            "multi_evidence_complete_rate@5": b1["oracle_ceilings"]["MultiComplete@5"],
        },
    }
    names = ("GoldR@5", "required_claim_coverage@5", "multi_evidence_complete_rate@5")
    report: dict[str, Any] = {"ceilings": ceilings, "actual_to_oracle_gap": {}}
    for cohort, rows in cohort_rows.items():
        report["actual_to_oracle_gap"][cohort] = {}
        for arm, arm_rows in rows.items():
            actual = metrics(arm_rows)[0]
            report["actual_to_oracle_gap"][cohort][arm] = {
                name: round(ceilings[cohort][name] - actual[name], 6) for name in names
            }
    combined_ceiling: dict[str, float] = {}
    for name in names:
        values: list[float] = []
        for cohort, rows in cohort_rows.items():
            count = len(rows["D2-R0"]) if name == "GoldR@5" else sum(
                item[name] is not None for item in rows["D2-R0"]
            )
            values.extend([ceilings[cohort][name]] * count)
        combined_ceiling[name] = round(sum(values) / len(values), 6)
    report["ceilings"]["COMBINED236"] = combined_ceiling
    report["actual_to_oracle_gap"]["COMBINED236"] = {}
    for arm, arm_rows in all_rows.items():
        actual = metrics(arm_rows)[0]
        report["actual_to_oracle_gap"]["COMBINED236"][arm] = {
            name: round(combined_ceiling[name] - actual[name], 6) for name in names
        }
    return report


def main() -> None:
    load_dotenv(CANONICAL_ROOT / ".env", override=True)
    settings = Settings()
    provider = build_llm_provider(settings)
    checkpoint = load_checkpoint()
    cohort_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    paired: dict[str, int] = {}
    selector_records: list[dict[str, Any]] = []
    for cohort, snapshot, trace, questions, gold_map in (c1_cohort(), b1_cohort()):
        trace_by_question = {item["question_id"]: item for item in trace["records"]}
        rows = {arm: [] for arm in ARMS}
        for record in snapshot["records"]:
            question = questions[record["question_id"]]
            trace_record = trace_by_question[question["id"]]
            by_id = {candidate["candidate_unit_id"]: candidate for candidate in record["candidates"]}
            reranker_ids = trace_record["candidate_unit_ids"]
            ordered = [by_id[item] for item in reranker_ids]
            if len(ordered) != CANDIDATE_DEPTH or len(by_id) != CANDIDATE_DEPTH:
                raise RuntimeError(f"D2_CANDIDATE_DEPTH_VIOLATION:{cohort}:{question['id']}")
            selected_ids, selector_record = select_or_fail_closed(
                provider=provider,
                cohort=cohort,
                question=question,
                ordered=ordered,
                scores=trace_record["rerank_scores"],
                checkpoint=checkpoint,
            )
            selector_records.append(selector_record)
            selected_order = ordered_with_selected_first(ordered, selected_ids)
            if set(item["candidate_unit_id"] for item in ordered) != set(item["candidate_unit_id"] for item in selected_order):
                raise RuntimeError(f"D2_PAIRED_IDENTITY_VIOLATION:{cohort}:{question['id']}")
            base_score = score_question(question, ordered, gold_map)
            selector_score = score_question(question, selected_order, gold_map)
            selector_score["selected_candidate_ids"] = selected_ids
            selector_score["selector_status"] = selector_record["status"]
            rows["D2-R0"].append(base_score)
            rows["D2-R1"].append(selector_score)
        paired[cohort] = len(rows["D2-R0"])
        cohort_rows[cohort] = rows
    all_rows = {arm: [*cohort_rows["DEV176"][arm], *cohort_rows["POSTBLIND60"][arm]] for arm in ARMS}
    cohorts = {
        name: {arm: {"metrics": metrics(rows[arm])[0], "losses": metrics(rows[arm])[1]} for arm in ARMS}
        for name, rows in {**cohort_rows, "COMBINED236": all_rows}.items()
    }
    safety_report = {name: safety(rows["D2-R0"], rows["D2-R1"]) for name, rows in cohort_rows.items()}
    oracle = oracle_gaps(cohort_rows, all_rows)
    gate = True
    for name in ("DEV176", "POSTBLIND60"):
        baseline = cohorts[name]["D2-R0"]["metrics"]
        candidate = cohorts[name]["D2-R1"]["metrics"]
        gate &= all(candidate[key] - baseline[key] >= -0.02 for key in ("GoldR@5", "MRR", "NDCG@10", "context_precision", "context_recall", "required_claim_coverage@5", "multi_evidence_complete_rate@5"))
        gate &= safety_report[name]["pass"]
    combined_base = cohorts["COMBINED236"]["D2-R0"]["metrics"]
    combined_candidate = cohorts["COMBINED236"]["D2-R1"]["metrics"]
    gate &= all(combined_candidate[key] - combined_base[key] >= -0.02 for key in ("GoldR@5", "MRR", "NDCG@10"))
    gate &= combined_candidate["required_claim_coverage@5"] >= combined_base["required_claim_coverage@5"]
    b1_snapshot = load_json(ROOT / "artifacts/rag-quality-v3/b1/execution/snapshots/b1-candidate-snapshot-v1.json")
    b1_trace = load_json(ROOT / "artifacts/rag-quality-v3/b1/execution/traces/b1-r1-ordering-v1.json")
    recovery = residual_recovery(cohort_rows["POSTBLIND60"], b1_snapshot, b1_trace)
    set_fixed = recovery.get("SET_COMPLETENESS", {}).get("fixed", 0)
    gate &= combined_candidate["multi_evidence_complete_rate@5"] >= combined_base["multi_evidence_complete_rate@5"] and (combined_candidate["multi_evidence_complete_rate@5"] - combined_base["multi_evidence_complete_rate@5"] >= 0.05 or set_fixed >= 3)
    statuses = defaultdict(int)
    usage = defaultdict(int)
    calls = 0
    for record in selector_records:
        statuses[record["status"]] += 1
        calls += int(record.get("request_attempt_count", 0))
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            usage[key] += int(record.get("usage", {}).get(key, 0))
    selection_snapshot = {
        "schema_version": "ragq3-d2-selector-snapshot-v1",
        "records": selector_records,
        "global_sha256": sha(selector_records),
    }
    save_json(OUT / "d2-r1-selector-snapshot-v1.json", selection_snapshot)
    result = {
        "schema_version": "ragq3-d2-final-decision-v1",
        "freeze_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "provider": {"name": settings.llm_provider_name or settings.llm_provider, "model": settings.llm_model, "base_url": settings.llm_base_url, "temperature": settings.llm_temperature, "max_output_tokens": 128, "max_retries": settings.llm_max_retries},
        "selector_calls": {"calls": calls, "success": statuses["SUCCESS"], "fail_closed_invalid_output": statuses["FAIL_CLOSED_INVALID_SELECTOR_OUTPUT"], "failures": statuses["HARD_PROVIDER_ERROR"], "usage": dict(usage)},
        "paired_identity": paired,
        "cohorts": cohorts,
        "oracle": oracle,
        "d0_d1_residual_recovery": recovery,
        "safety": safety_report,
        "frozen_gate": {"D2-R1": {"PASS": bool(gate)}},
        "selected_candidate": "D2-R1" if gate else "NONE",
        "decision": "LISTWISE_EVIDENCE_SET_SELECTION_VALIDATED" if gate else "POST_RETRIEVAL_SELECTION_PROGRAM_CLOSED_NO_PROMOTION",
        "B2_FRESH_BLIND_ELIGIBLE": "yes" if gate else "no",
        "post_retrieval_optimization_closed": "no" if gate else "yes",
        "invariants": {"retrieval": 0, "embedding": 0, "reranker": 0, "full_qa": "NOT_RUN", "production_change": "no", "tuning": "no", "candidate_membership": "identical"},
        "selection_snapshot_sha256": selection_snapshot["global_sha256"],
    }
    save_json(FINAL, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
