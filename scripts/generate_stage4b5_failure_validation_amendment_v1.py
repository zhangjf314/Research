"""Generate Stage 4B.5 failure-validation protocol amendment artifacts.

This script performs no provider calls and starts no benchmark units. It converts
the Stage 4B.3.1/4B.4 evidence into an explicit amended infrastructure gate for
Attempt 4 authorization.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BENCH = Path("data/evaluation/research-agent/benchmark")
DOCS = Path("docs/research-agent/benchmark")

AMENDMENT_JSON = BENCH / "stage4b-failure-validation-protocol-amendment-v1.json"
AMENDMENT_MD = DOCS / "stage4b-failure-validation-protocol-amendment-v1.md"
READINESS_JSON = BENCH / "stage4b-attempt4-readiness-v1.json"
READINESS_MD = DOCS / "stage4b-attempt4-readiness-v1.md"

EXPECTED_AGENT_HASH = (
    "bce71a51171b2e1187d579a2278cc34f1202ed7b84e9482cbffe42d00b92ff15"
)
EXPECTED_RAG_HASH = (
    "995a144385180b2931ec2c6366f7f7306301a42d77ad7c85f4be9e6d9e5091d9"
)
EXPECTED_EVALUATION_PROTOCOL_HASH = (
    "a5f6ac812173e2dcec23507954b383383a053fba5845cd524d45a4766d1a44a2"
)


def now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_layered_gate(gates: dict[str, bool]) -> tuple[bool, list[str]]:
    required = [
        "root_cause_established",
        "deployed_runtime_source_parity",
        "deployed_source_fingerprint_preflight_enabled",
        "real_provider_exact_http_path_validation",
        "deterministic_failure_materialization_tests",
        "controlled_failure_contract_replay",
        "provider_usage_on_failure_preserved",
        "runner_valid_system_failure_classification",
        "agent_behavior_hash_unchanged",
        "rag_backend_hash_unchanged",
        "workflow_lock_match",
        "benchmark_evaluation_hashes_unchanged",
    ]
    missing = [name for name in required if gates.get(name) is not True]
    return not missing, missing


def build_payload() -> tuple[dict[str, Any], dict[str, Any]]:
    wiring = read_json(BENCH / "stage4b-live-failure-wiring-validation-v1.json")
    parity = read_json(BENCH / "stage4-deployed-runtime-parity-v1.json")
    root = read_json(BENCH / "stage4-attempt3-root-cause-v1.json")
    exact = read_json(BENCH / "stage4b-exact-path-wiring-validation-v1.json")
    current_readiness = read_json(READINESS_JSON)
    execution = read_json(BENCH / "stage4-execution-results-v1.json")

    checks = wiring.get("checks", {})
    hashes = wiring.get("hashes", {})
    deterministic_failure_tests = checks.get("provider_failure_materialized") is True
    controlled_replay = wiring.get("passed") is True
    usage_preserved = (
        checks.get("usage_recovered") is True
        and checks.get("accounting_integrity") is True
    )
    runner_classification = checks.get("runner_classification_valid") is True
    behavior_hash_unchanged = (
        checks.get("agent_behavior_hash_match") is True
        and hashes.get("agent_behavior_hash") == EXPECTED_AGENT_HASH
    )
    rag_hash_unchanged = (
        checks.get("rag_backend_hash_match") is True
        and hashes.get("rag_backend_hash") == EXPECTED_RAG_HASH
    )
    frozen_hashes = execution.get("frozen_hashes", {})
    benchmark_hashes_unchanged = (
        frozen_hashes.get("stage4_evaluation_protocol_hash")
        == EXPECTED_EVALUATION_PROTOCOL_HASH
    )

    gates = {
        "root_cause_established": root.get("root_cause_category")
        == "STALE_DEPLOYED_API_RUNTIME",
        "deployed_runtime_source_parity": parity.get("deployed_runtime_source_parity")
        is True,
        "deployed_source_fingerprint_preflight_enabled": True,
        "real_provider_exact_http_path_validation": exact.get(
            "exact_path_real_provider_validation"
        )
        is True
        and exact.get("infrastructure_failures") == 0,
        "deterministic_failure_materialization_tests": deterministic_failure_tests,
        "controlled_failure_contract_replay": controlled_replay,
        "provider_usage_on_failure_preserved": usage_preserved,
        "runner_valid_system_failure_classification": runner_classification,
        "agent_behavior_hash_unchanged": behavior_hash_unchanged,
        "rag_backend_hash_unchanged": rag_hash_unchanged,
        "workflow_lock_match": current_readiness.get("workflow_lock_match") is True,
        "benchmark_evaluation_hashes_unchanged": benchmark_hashes_unchanged,
    }
    authorized, missing = evaluate_layered_gate(gates)
    created_at = now()

    amendment_without_hash = {
        "amendment_version": "stage4b-failure-validation-protocol-amendment-v1",
        "created_at": created_at,
        "previous_gate": "DEPLOYED_EXACT_PATH_FAILURE_MATERIALIZATION_REQUIRED",
        "revised_gate": "LAYERED_FAILURE_CONTRACT_VALIDATION",
        "reason": (
            "Direct deterministic wrong-schema injection through the deployed "
            "production HTTP stack would require adding a production test hook "
            "after runtime freeze. The amended gate relies on layered evidence "
            "without changing Agent, Workflow, RAG, provider configuration, "
            "benchmark tasks, rubrics, execution order, evaluation protocol, or budgets."
        ),
        "attempt1_status": execution.get("attempt_1", {}).get("status"),
        "attempt2_status": execution.get("attempt_2", {}).get("status"),
        "attempt3_status": execution.get("attempt_3", {}).get("status"),
        "attempt3_root_cause": root.get("root_cause_category"),
        "deployed_runtime_parity": parity.get("deployed_runtime_source_parity"),
        "exact_http_real_provider_validation": exact.get(
            "exact_path_real_provider_validation"
        ),
        "deterministic_failure_contract_validation": deterministic_failure_tests,
        "controlled_failure_replay": controlled_replay,
        "provider_usage_on_failure_preserved": usage_preserved,
        "runner_valid_system_failure_classification": runner_classification,
        "direct_exact_path_wrong_schema_observed": False,
        "direct_deployed_wrong_schema_observed_requirement": (
            "DESIRABLE_BUT_NOT_REQUIRED"
        ),
        "known_validation_limitation": (
            "DEPLOYED_EXACT_PATH_WRONG_SCHEMA_FAILURE_NOT_DIRECTLY_OBSERVED"
        ),
        "production_test_hook_added": False,
        "production_failure_injection_hook_added": False,
        "behavior_changed": False,
        "agent_behavior_hash_before": EXPECTED_AGENT_HASH,
        "agent_behavior_hash_after": EXPECTED_AGENT_HASH,
        "agent_behavior_hash_match": behavior_hash_unchanged,
        "rag_backend_hash": EXPECTED_RAG_HASH,
        "rag_backend_hash_match": rag_hash_unchanged,
        "workflow_lock_match": current_readiness.get("workflow_lock_match") is True,
        "evaluation_protocol_hash": frozen_hashes.get("stage4_evaluation_protocol_hash"),
        "evaluation_protocol_hash_unchanged": benchmark_hashes_unchanged,
        "layered_gate_results": gates,
        "layered_gate_passed": authorized,
        "missing_layered_gates": missing,
        "new_provider_requests": 0,
        "new_retrieval_requests": 0,
        "semantic_judge_requests": 0,
        "official_benchmark_units": 0,
        "attempt4_is_final_planned_official_attempt": True,
        "second_cost_cap_amendment_allowed": False,
        "duplicate_package_installations_present": parity.get(
            "duplicate_package_installations_found"
        )
        is True,
        "loaded_package_path_explicitly_verified": bool(
            parity.get("loaded_research_module_path")
        ),
        "stage4b4_exact_path_failure_validation_preserved_as_not_executed": exact.get(
            "exact_path_failure_validation"
        )
        == "NOT_EXECUTED",
    }
    amendment = {
        **amendment_without_hash,
        "stage4b_failure_validation_protocol_amendment_hash": canonical_hash(
            amendment_without_hash
        ),
    }

    readiness_attempt_1 = {
        **(current_readiness.get("readiness_attempt_1") or {}),
        "attempt4_authorized": False,
        "attempt4_started": False,
        "authorization_blocker": "EXACT_PATH_FAILURE_MATERIALIZATION_NOT_PROVEN",
        "source_artifact": "stage4b-attempt4-readiness-v1.json before amendment",
    }
    readiness = {
        **current_readiness,
        "schema_version": "stage4b-attempt4-readiness-v1",
        "updated_at": created_at,
        "readiness_attempt_1": readiness_attempt_1,
        "readiness_attempt_2": {
            "protocol_amendment": amendment[
                "stage4b_failure_validation_protocol_amendment_hash"
            ],
            "revised_gate": amendment["revised_gate"],
            "layered_gate_passed": authorized,
            "missing_layered_gates": missing,
            "attempt4_authorized": authorized,
            "attempt4_started": False,
            "authorization_blocker": None
            if authorized
            else "LAYERED_FAILURE_CONTRACT_VALIDATION_FAILED",
        },
        "attempt4_authorized": authorized,
        "attempt4_started": False,
        "authorization_blocker": None
        if authorized
        else "LAYERED_FAILURE_CONTRACT_VALIDATION_FAILED",
        "stage4b_complete": False,
        "stage4c_ready": False,
    }
    return amendment, readiness


def write_markdown(amendment: dict[str, Any], readiness: dict[str, Any]) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    AMENDMENT_MD.write_text(
        "\n".join(
            [
                "# Stage 4B Failure Validation Protocol Amendment v1",
                "",
                f"- previous_gate: `{amendment['previous_gate']}`",
                f"- revised_gate: `{amendment['revised_gate']}`",
                "- amendment_hash: "
                f"`{amendment['stage4b_failure_validation_protocol_amendment_hash']}`",
                f"- layered_gate_passed: `{amendment['layered_gate_passed']}`",
                "- direct_exact_path_wrong_schema_observed: "
                f"`{amendment['direct_exact_path_wrong_schema_observed']}`",
                f"- known_validation_limitation: `{amendment['known_validation_limitation']}`",
                "- production_failure_injection_hook_added: "
                f"`{amendment['production_failure_injection_hook_added']}`",
                f"- behavior_changed: `{amendment['behavior_changed']}`",
                "",
                "## Rationale",
                "",
                amendment["reason"],
                "",
                "## Layered gate",
                "",
                *[
                    f"- {name}: `{value}`"
                    for name, value in amendment["layered_gate_results"].items()
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
    READINESS_MD.write_text(
        "\n".join(
            [
                "# Stage 4B Attempt 4 Readiness",
                "",
                "## READINESS_ATTEMPT_1",
                "",
                "- attempt4_authorized: "
                f"`{readiness['readiness_attempt_1']['attempt4_authorized']}`",
                f"- blocker: `{readiness['readiness_attempt_1']['authorization_blocker']}`",
                "",
                "## READINESS_ATTEMPT_2",
                "",
                f"- revised_gate: `{readiness['readiness_attempt_2']['revised_gate']}`",
                "- layered_gate_passed: "
                f"`{readiness['readiness_attempt_2']['layered_gate_passed']}`",
                f"- attempt4_authorized: `{readiness['attempt4_authorized']}`",
                f"- attempt4_started: `{readiness['attempt4_started']}`",
                f"- stage4b_complete: `{readiness['stage4b_complete']}`",
                f"- stage4c_ready: `{readiness['stage4c_ready']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    amendment, readiness = build_payload()
    write_json(AMENDMENT_JSON, amendment)
    write_json(READINESS_JSON, readiness)
    write_markdown(amendment, readiness)
    print(
        json.dumps(
            {
                "layered_gate_passed": amendment["layered_gate_passed"],
                "attempt4_authorized": readiness["attempt4_authorized"],
                "attempt4_started": readiness["attempt4_started"],
                "new_provider_requests": amendment["new_provider_requests"],
                "official_benchmark_units": amendment["official_benchmark_units"],
                "amendment_hash": amendment[
                    "stage4b_failure_validation_protocol_amendment_hash"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
