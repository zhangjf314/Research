"""D2B frozen listwise evaluation; sole delta from D2 is output budget 256."""
# ruff: noqa
from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from paper_research.config import Settings
from paper_research.providers.factory import build_llm_provider
from paper_research.providers.llm import LLMProviderError
from run_c2_snapshot_native_reranking import metrics, score_question
from run_d2_listwise_evaluation import (
    CANONICAL_ROOT,
    ROOT,
    SYSTEM_PROMPT,
    b1_cohort,
    c1_cohort,
    oracle_gaps,
    ordered_with_selected_first,
    residual_recovery,
    safety,
    selector_user_prompt,
    sha,
    validate_selection,
)

OUT = ROOT / "artifacts/rag-quality-v3/d2b/execution"
CHECKPOINT = OUT / "selector-checkpoint-v1.jsonl"
FINAL = OUT / "d2b-final-decision-v1.json"
ARMS = ("D2B-R0", "D2B-R1")
MAX_OUTPUT_TOKENS = 256
CANDIDATE_DEPTH = 20
SELECTED_COUNT = 5


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_checkpoint() -> dict[str, dict[str, Any]]:
    if not CHECKPOINT.exists():
        return {}
    records = {}
    for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        records[item["key"]] = item
    return records


def append_checkpoint(record: dict[str, Any]) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def select(provider: Any, cohort: str, question: dict[str, Any], ordered: list[dict[str, Any]], scores: dict[str, float], checkpoint: dict[str, dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    key = f"{cohort}:{question['id']}"
    allowed = [item["candidate_unit_id"] for item in ordered]
    input_sha = sha({"question": question["query"], "ids": allowed, "scores": scores})
    existing = checkpoint.get(key)
    if existing and existing.get("input_sha256") == input_sha:
        return list(existing["selected_candidate_ids"]), existing
    try:
        response = provider.generate_structured_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=selector_user_prompt(question, ordered, scores),
            schema_name="ragq3-d2b-listwise-selector-v1",
            request_context={"task_id": f"d2b-{cohort}-{question['id']}", "run_id": "ragq3-d2b"},
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        selected = validate_selection(response.payload, allowed)
        record = {
            "key": key, "cohort": cohort, "question_id": question["id"], "input_sha256": input_sha,
            "selected_candidate_ids": selected, "status": "VALID", "finish_reason": "non_length",
            "provider": response.provider, "model": response.model,
            "request_attempt_count": response.request_attempt_count, "retry_count": response.retry_count,
            "latency_ms": response.total_latency_ms, "usage": response.usage.model_dump(),
            "response_sha256": sha(response.payload),
        }
    except LLMProviderError as exc:
        finish = "length" if "finish_reason:length" in exc.retry_reasons else "provider_or_schema_error"
        record = {
            "key": key, "cohort": cohort, "question_id": question["id"], "input_sha256": input_sha,
            "selected_candidate_ids": allowed[:SELECTED_COUNT], "status": "INVALID",
            "finish_reason": finish, "error_code": exc.error_code, "stage": exc.stage,
            "request_attempt_count": exc.api_request_count, "retry_reasons": exc.retry_reasons,
        }
    except (TypeError, ValueError) as exc:
        record = {
            "key": key, "cohort": cohort, "question_id": question["id"], "input_sha256": input_sha,
            "selected_candidate_ids": allowed[:SELECTED_COUNT], "status": "INVALID",
            "finish_reason": "schema_invalid", "error_code": type(exc).__name__, "request_attempt_count": 1,
        }
    append_checkpoint(record)
    return list(record["selected_candidate_ids"]), record


def main() -> None:
    load_dotenv(CANONICAL_ROOT / ".env", override=True)
    settings = Settings()
    provider = build_llm_provider(settings)
    checkpoint = load_checkpoint()
    cohort_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    paired: dict[str, int] = {}
    selector_records: list[dict[str, Any]] = []
    for cohort, snapshot, trace, questions, gold_map in (c1_cohort(), b1_cohort()):
        traces = {item["question_id"]: item for item in trace["records"]}
        rows = {arm: [] for arm in ARMS}
        for source in snapshot["records"]:
            question = questions[source["question_id"]]
            trace_record = traces[question["id"]]
            by_id = {item["candidate_unit_id"]: item for item in source["candidates"]}
            ordered = [by_id[item] for item in trace_record["candidate_unit_ids"]]
            if len(ordered) != CANDIDATE_DEPTH or len(by_id) != CANDIDATE_DEPTH:
                raise RuntimeError(f"D2B_CANDIDATE_DEPTH_VIOLATION:{cohort}:{question['id']}")
            selected, record = select(provider, cohort, question, ordered, trace_record["rerank_scores"], checkpoint)
            selected_order = ordered_with_selected_first(ordered, selected)
            if set(item["candidate_unit_id"] for item in ordered) != set(item["candidate_unit_id"] for item in selected_order):
                raise RuntimeError(f"D2B_PAIRED_IDENTITY_VIOLATION:{cohort}:{question['id']}")
            scored = score_question(question, selected_order, gold_map)
            scored["selected_candidate_ids"] = selected
            scored["selector_status"] = record["status"]
            rows["D2B-R0"].append(score_question(question, ordered, gold_map))
            rows["D2B-R1"].append(scored)
            selector_records.append(record)
        paired[cohort] = len(rows["D2B-R0"])
        cohort_rows[cohort] = rows
    statuses = Counter(record["status"] for record in selector_records)
    finish_reasons = Counter(record.get("finish_reason", "unknown") for record in selector_records)
    execution_valid = statuses["VALID"] == 236 and statuses["INVALID"] == 0
    snapshot = {"schema_version": "ragq3-d2b-selector-snapshot-v1", "records": selector_records, "global_sha256": sha(selector_records)}
    save_json(OUT / "d2b-r1-selector-snapshot-v1.json", snapshot)
    result: dict[str, Any] = {
        "schema_version": "ragq3-d2b-final-decision-v1",
        "freeze_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "historical_d2": {"decision": "POST_RETRIEVAL_SELECTION_PROGRAM_CLOSED_NO_PROMOTION", "selector_quality": "NOT_EVALUATED", "reason": "236/236 finish_reason=length; valid outputs=0; fail-closed R1=R0"},
        "provider": {"name": settings.llm_provider_name or settings.llm_provider, "model": settings.llm_model, "temperature": settings.llm_temperature, "max_output_tokens": MAX_OUTPUT_TOKENS},
        "selector_calls": {"calls": len(selector_records), "valid": statuses["VALID"], "invalid": statuses["INVALID"], "finish_reasons": dict(finish_reasons)},
        "paired_identity": paired,
        "execution_valid": execution_valid,
        "invariants": {"retrieval": 0, "embedding": 0, "reranker": 0, "full_qa": "NOT_RUN", "production_change": "no", "candidate_membership": "identical"},
        "selection_snapshot_sha256": snapshot["global_sha256"],
    }
    if not execution_valid:
        result.update({
            "quality": "NOT_EVALUATED",
            "frozen_gate": {"D2B-R1": {"PASS": False, "reason": "D2B_EXECUTION_NOT_VALID"}},
            "selected_candidate": "NONE", "decision": "POST_RETRIEVAL_OPTIMIZATION_CLOSED",
            "B2_FRESH_BLIND_ELIGIBLE": "no", "post_retrieval_optimization_closed": "yes",
        })
    else:
        all_rows = {arm: [*cohort_rows["DEV176"][arm], *cohort_rows["POSTBLIND60"][arm]] for arm in ARMS}
        cohorts = {
            name: {arm: {"metrics": metrics(rows[arm])[0], "losses": metrics(rows[arm])[1]} for arm in ARMS}
            for name, rows in {**cohort_rows, "COMBINED236": all_rows}.items()
        }
        safety_report = {name: safety(rows["D2B-R0"], rows["D2B-R1"]) for name, rows in cohort_rows.items()}
        gate = True
        for name in ("DEV176", "POSTBLIND60"):
            baseline, candidate = cohorts[name]["D2B-R0"]["metrics"], cohorts[name]["D2B-R1"]["metrics"]
            gate &= all(candidate[key] - baseline[key] >= -0.02 for key in ("GoldR@5", "MRR", "NDCG@10", "context_precision", "context_recall", "required_claim_coverage@5", "multi_evidence_complete_rate@5"))
            gate &= safety_report[name]["pass"]
        baseline, candidate = cohorts["COMBINED236"]["D2B-R0"]["metrics"], cohorts["COMBINED236"]["D2B-R1"]["metrics"]
        gate &= all(candidate[key] - baseline[key] >= -0.02 for key in ("GoldR@5", "MRR", "NDCG@10"))
        gate &= candidate["required_claim_coverage@5"] >= baseline["required_claim_coverage@5"]
        b1_snapshot = load_json(ROOT / "artifacts/rag-quality-v3/b1/execution/snapshots/b1-candidate-snapshot-v1.json")
        b1_trace = load_json(ROOT / "artifacts/rag-quality-v3/b1/execution/traces/b1-r1-ordering-v1.json")
        recovery = residual_recovery(
            {"D2-R1": cohort_rows["POSTBLIND60"]["D2B-R1"]},
            b1_snapshot,
            b1_trace,
        )
        gate &= candidate["multi_evidence_complete_rate@5"] >= baseline["multi_evidence_complete_rate@5"] and (
            candidate["multi_evidence_complete_rate@5"] - baseline["multi_evidence_complete_rate@5"] >= 0.05
            or recovery.get("SET_COMPLETENESS", {}).get("fixed", 0) >= 3
        )
        result.update({
            "quality": "EVALUATED",
            "cohorts": cohorts,
            "oracle": oracle_gaps(
                {
                    cohort: {"D2-R0": rows["D2B-R0"], "D2-R1": rows["D2B-R1"]}
                    for cohort, rows in cohort_rows.items()
                },
                {"D2-R0": all_rows["D2B-R0"], "D2-R1": all_rows["D2B-R1"]},
            ),
            "safety": safety_report,
            "d0_d1_residual_recovery": recovery,
            "frozen_gate": {"D2B-R1": {"PASS": bool(gate)}},
            "selected_candidate": "D2B-R1" if gate else "NONE",
            "decision": "LISTWISE_EVIDENCE_SET_SELECTION_VALIDATED" if gate else "POST_RETRIEVAL_OPTIMIZATION_CLOSED",
            "B2_FRESH_BLIND_ELIGIBLE": "yes" if gate else "no",
            "post_retrieval_optimization_closed": "no" if gate else "yes",
        })
    save_json(FINAL, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
