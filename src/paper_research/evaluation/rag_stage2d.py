from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DECISION_VALUES = {
    "SELECTED",
    "REJECTED",
    "SUPPORTED_BUT_UNRESOLVED",
    "BASELINE_RETAINED",
    "NOT_APPLICABLE",
}


def canonical_json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_stage2_artifacts(
    stage2a: dict[str, Any], stage2b: dict[str, Any], stage2c: dict[str, Any]
) -> dict[str, Any]:
    issues = []
    for name, artifact in (
        ("stage2a", stage2a),
        ("stage2b", stage2b),
        ("stage2c", stage2c),
    ):
        if artifact.get("split") != "dev":
            issues.append(f"{name}_split_not_dev")
        if int(artifact.get("test_questions_evaluated") or 0) != 0:
            issues.append(f"{name}_test_questions_evaluated_nonzero")
        if artifact.get("test_protocol_violation"):
            issues.append(f"{name}_test_protocol_violation")
    return {
        "valid": not issues,
        "issues": issues,
        "stage2_test_optimization_runs": sum(
            int(artifact.get("test_questions_evaluated") or 0)
            for artifact in (stage2a, stage2b, stage2c)
        ),
    }


def behavioral_config_projection(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "retrieval": config["retrieval"],
        "reranker": config["reranker"],
        "query_rewrite": config["query_rewrite"],
        "query_decomposition": config["query_decomposition"],
        "context_selection": config["context_selection"],
        "generation": config["generation"],
        "embedding": config["embedding"],
    }


def component_decisions() -> dict[str, str]:
    return {
        "dense_only": "REJECTED",
        "sparse_only": "REJECTED",
        "current_hybrid": "SELECTED",
        "lexical_rerank": "REJECTED",
        "single_rewrite": "REJECTED",
        "original_plus_rewrite": "REJECTED",
        "query_decomposition": "REJECTED",
        "baseline_context": "BASELINE_RETAINED",
        "score_budgeted_dedup_context": "REJECTED",
        "diversity_aware_context": "REJECTED",
        "context_selection_bottleneck": "SUPPORTED_BUT_UNRESOLVED",
    }


def validate_component_decisions(decisions: dict[str, str]) -> None:
    invalid = {key: value for key, value in decisions.items() if value not in DECISION_VALUES}
    if invalid:
        raise ValueError(f"invalid component decisions: {invalid}")


def final_config_from_baseline(
    baseline_config: dict[str, Any], decisions: dict[str, str]
) -> dict[str, Any]:
    return {
        "schema_version": "rag-stage2-final-config-v1",
        "retrieval": {
            "mode": "hybrid",
            "selected": decisions["current_hybrid"] == "SELECTED",
            "recall_k": baseline_config["retrieval"]["recall_k"],
            "top_k": baseline_config["retrieval"]["top_k"],
            "score_threshold": baseline_config["retrieval"]["score_threshold"],
            "production_collection": baseline_config["retrieval"]["production_collection"],
        },
        "embedding": baseline_config["embedding"],
        "reranker": {
            "enabled": False,
            "provider": baseline_config["reranker"]["provider"],
            "model": baseline_config["reranker"]["model"],
            "reason": "stage2a_gate_failed",
        },
        "query_rewrite": {
            "enabled": False,
            "reason": "stage2b_gate_failed",
        },
        "query_decomposition": {
            "enabled": False,
            "reason": "stage2b_gate_failed",
        },
        "context_selection": {
            "mode": "baseline",
            "experimental_selector_enabled": False,
            "bottleneck_confirmed": True,
            "reason": "stage2c_offline_gate_failed",
        },
        "generation": baseline_config["generation"],
        "stage2_final_behavior_change": False,
    }


def metric_row(
    label: str,
    metrics: dict[str, Any],
    decision: str,
    *,
    required_claim_metric: float | None = None,
) -> dict[str, Any]:
    return {
        "configuration": label,
        "recall_at_10": metrics.get("recall_at_10"),
        "mrr_at_10": metrics.get("mrr_at_10"),
        "ndcg_at_10": metrics.get("ndcg_at_10"),
        "evidence_coverage_at_10": metrics.get("evidence_coverage_at_10"),
        "required_claim_evidence_coverage_at_10": required_claim_metric,
        "decision": decision,
    }
