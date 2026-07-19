from __future__ import annotations

import json
import subprocess

from scripts import production_full_qa_gate_v1 as gate


def clear_budget_env(monkeypatch):
    for name in gate.FULL_QA_BUDGET_VARS + gate.DEEP_RESEARCH_BUDGET_VARS:
        monkeypatch.delenv(name, raising=False)


def test_live_calls_disabled_blocks_full_qa(monkeypatch):
    clear_budget_env(monkeypatch)
    monkeypatch.setattr(gate, "_env_value", lambda name: None)

    status = gate.budget_status()

    assert status["status"] == "BLOCKED_BY_LIVE_MODEL_CALLS_DISABLED"
    assert status["live_model_calls_enabled"] is False
    assert status["full_qa_budget_ready"] is False
    assert status["smoke_allowed"] is False
    assert "FULL_QA_MAX_COST_USD" in status["missing_full_qa_budget_vars"]


def test_live_enabled_without_budget_allows_smoke_only(monkeypatch):
    clear_budget_env(monkeypatch)
    monkeypatch.setattr(
        gate,
        "_env_value",
        lambda name: "true" if name == "LIVE_MODEL_CALLS_ENABLED" else None,
    )

    status = gate.budget_status()

    assert status["status"] == "SMOKE_ONLY_BUDGET_INCOMPLETE"
    assert status["live_model_calls_enabled"] is True
    assert status["smoke_allowed"] is True
    assert status["full_qa_budget_ready"] is False
    assert "FULL_QA_MAX_INPUT_TOKENS" in status["missing_full_qa_budget_vars"]
    assert "FULL_QA_MAX_COST_USD" in status["missing_full_qa_budget_vars"]


def test_full_qa_budget_ready_when_all_required_values_exist(monkeypatch):
    clear_budget_env(monkeypatch)
    values = {
        "LIVE_MODEL_CALLS_ENABLED": "true",
        "FULL_QA_MAX_ITEMS": "50",
        "FULL_QA_MAX_INPUT_TOKENS": "200000",
        "FULL_QA_MAX_OUTPUT_TOKENS": "50000",
        "FULL_QA_MAX_COST_USD": "5",
        "FULL_QA_MAX_TOTAL_SECONDS": "3600",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    status = gate.budget_status()

    assert status["status"] == "FULL_QA_BUDGET_READY"
    assert status["live_model_calls_enabled"] is True
    assert status["smoke_allowed"] is True
    assert status["full_qa_budget_ready"] is True
    assert status["missing_full_qa_budget_vars"] == []


def test_gold_dev_manifest_is_internal_dev_not_blind():
    manifest = gate.build_gold_manifest()

    assert manifest["dataset_id"] == "gold-dev-v1"
    assert manifest["role"] == "人工审核的内部开发评测集"
    assert manifest["blind"] is False
    assert manifest["total_count"] == 50
    assert manifest["approved_count"] == 50
    assert manifest["answerable_count"] == 48
    assert manifest["unanswerable_count"] == 2
    assert manifest["strong_generalization_claim_allowed"] is False
    assert len(manifest["dataset_sha256"]) == 64


def test_safe_secret_never_exposes_value():
    secret = "sk-test-secret-value"

    safe = gate._safe_secret(secret)

    assert safe["present"] is True
    assert safe["length"] == len(secret)
    assert safe["sha256_prefix"]
    assert secret not in str(safe)


def test_container_budget_status_blocks_missing_injected_vars(monkeypatch):
    def fake_check_output(*args, **kwargs):
        del args, kwargs
        return json.dumps({name: False for name in gate.CONTAINER_BUDGET_VARS})

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    status = gate.container_budget_status()

    assert status["status"] == "BLOCKED_BY_CONTAINER_LIVE_MODEL_CALLS_DISABLED"
    assert status["full_qa_budget_ready"] is False
    assert status["missing_full_qa_budget_vars"] == gate.FULL_QA_BUDGET_VARS


def test_compose_injects_stage13_30_budget_variables():
    compose = (gate.ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    for name in gate.CONTAINER_BUDGET_VARS:
        assert f"{name}:" in compose
