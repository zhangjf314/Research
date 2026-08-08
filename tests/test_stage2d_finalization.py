from __future__ import annotations

import pytest

from paper_research.evaluation.rag_stage2d import (
    behavioral_config_projection,
    canonical_json_hash,
    component_decisions,
    final_config_from_baseline,
    validate_component_decisions,
    validate_stage2_artifacts,
)
from scripts.finalize_rag_stage2_v1 import build_lock, build_stage3_lock


def _baseline_config() -> dict:
    return {
        "baseline_config_hash": "h",
        "retrieval": {
            "recall_k": 20,
            "top_k": 5,
            "score_threshold": 0.12,
            "production_collection": "papers_production_v1",
        },
        "embedding": {"provider": "jina", "model": "jina-embeddings-v5-text-small"},
        "reranker": {"provider": "lexical", "model": "lexical-v1"},
        "generation": {"prompt_version": "qa-production-v1", "temperature": 0.0},
    }


def _baseline() -> dict:
    return {
        "dataset_version": "rag-gold-v1",
        "dev_dataset_hash": "dev",
        "test_dataset_hash": "test",
    }


def test_artifact_validation_rejects_test_optimization_runs() -> None:
    stage2a = {"split": "dev", "test_questions_evaluated": 0}
    stage2b = {"split": "dev", "test_questions_evaluated": 1}
    stage2c = {"split": "dev", "test_questions_evaluated": 0}
    result = validate_stage2_artifacts(stage2a, stage2b, stage2c)
    assert result["valid"] is False
    assert "stage2b_test_questions_evaluated_nonzero" in result["issues"]


def test_component_decisions_use_allowed_values() -> None:
    decisions = component_decisions()
    validate_component_decisions(decisions)
    assert decisions["current_hybrid"] == "SELECTED"
    assert decisions["context_selection_bottleneck"] == "SUPPORTED_BUT_UNRESOLVED"


def test_invalid_component_decision_fails() -> None:
    decisions = component_decisions()
    decisions["current_hybrid"] = "BEST_SCORE"
    with pytest.raises(ValueError):
        validate_component_decisions(decisions)


def test_rejected_components_do_not_enter_final_config() -> None:
    final_config = final_config_from_baseline(_baseline_config(), component_decisions())
    assert final_config["reranker"]["enabled"] is False
    assert final_config["query_rewrite"]["enabled"] is False
    assert final_config["query_decomposition"]["enabled"] is False
    assert final_config["context_selection"]["mode"] == "baseline"


def test_behavioral_config_hash_is_canonical() -> None:
    config = final_config_from_baseline(_baseline_config(), component_decisions())
    first = canonical_json_hash(behavioral_config_projection(config))
    second = canonical_json_hash(behavioral_config_projection(dict(reversed(config.items()))))
    assert first == second


def test_stage3_lock_uses_stage2_frozen_backend() -> None:
    config = final_config_from_baseline(_baseline_config(), component_decisions())
    lock = build_stage3_lock(config, "hash", _baseline())
    assert lock["rag_backend"]["retrieval"] == "Current Hybrid"
    assert lock["rag_backend"]["reranker"] == "disabled"
    assert "query_rewrite" in lock["agent_must_not_enable"]


def test_final_lock_reuses_stage1_test_baseline() -> None:
    lock = build_lock(_baseline(), _baseline_config(), component_decisions(), "hash")
    assert lock["stage2_final_test_source"] == "REUSED_STAGE1_FROZEN_BASELINE"
    assert lock["new_test_questions_evaluated"] == 0


def test_behavioral_projection_ignores_non_behavior_metadata() -> None:
    config = final_config_from_baseline(_baseline_config(), component_decisions())
    with_meta = dict(config)
    with_meta["created_at"] = "later"
    assert behavioral_config_projection(config) == behavioral_config_projection(with_meta)
