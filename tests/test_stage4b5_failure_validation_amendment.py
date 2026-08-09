from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path("scripts/generate_stage4b5_failure_validation_amendment_v1.py")


def _module():
    spec = importlib.util.spec_from_file_location("stage4b5_amendment", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _passing_gates() -> dict[str, bool]:
    return {
        "root_cause_established": True,
        "deployed_runtime_source_parity": True,
        "deployed_source_fingerprint_preflight_enabled": True,
        "real_provider_exact_http_path_validation": True,
        "deterministic_failure_materialization_tests": True,
        "controlled_failure_contract_replay": True,
        "provider_usage_on_failure_preserved": True,
        "runner_valid_system_failure_classification": True,
        "agent_behavior_hash_unchanged": True,
        "rag_backend_hash_unchanged": True,
        "workflow_lock_match": True,
        "benchmark_evaluation_hashes_unchanged": True,
    }


def test_layered_gate_does_not_require_direct_deployed_wrong_schema() -> None:
    module = _module()

    authorized, missing = module.evaluate_layered_gate(_passing_gates())

    assert authorized is True
    assert missing == []


def test_layered_gate_rejects_missing_required_evidence() -> None:
    module = _module()
    gates = _passing_gates()
    gates["deployed_runtime_source_parity"] = False

    authorized, missing = module.evaluate_layered_gate(gates)

    assert authorized is False
    assert missing == ["deployed_runtime_source_parity"]


def test_stage4b5_payload_preserves_failed_exact_path_history() -> None:
    module = _module()

    amendment, readiness = module.build_payload()

    assert amendment["attempt1_status"] == "INVALIDATED_INFRASTRUCTURE"
    assert amendment["attempt2_status"] == "INVALIDATED_INFRASTRUCTURE"
    assert amendment["attempt3_status"] == "INVALID"
    assert amendment["direct_exact_path_wrong_schema_observed"] is False
    assert (
        amendment["known_validation_limitation"]
        == "DEPLOYED_EXACT_PATH_WRONG_SCHEMA_FAILURE_NOT_DIRECTLY_OBSERVED"
    )
    assert (
        amendment["stage4b4_exact_path_failure_validation_preserved_as_not_executed"]
        is True
    )
    assert readiness["readiness_attempt_1"]["attempt4_authorized"] is False
    assert (
        readiness["readiness_attempt_1"]["authorization_blocker"]
        == "EXACT_PATH_FAILURE_MATERIALIZATION_NOT_PROVEN"
    )
    assert readiness["readiness_attempt_2"]["attempt4_authorized"] is True
    assert readiness["attempt4_started"] is False
    assert readiness["stage4b_complete"] is False
    assert readiness["stage4c_ready"] is False
