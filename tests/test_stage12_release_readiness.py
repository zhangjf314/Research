from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts.check_release_readiness_v1 import (
    EVIDENCE_KEYS,
    GATE_KEYS,
    ISSUE_KEYS,
    STAGE_KEYS,
    TASK_KEYS,
    evaluate,
)

ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_workspace(tmp_path: Path) -> Path:
    target = tmp_path / "Windows Style Workspace"
    for directory in ("data", "docs", "src", "tests"):
        shutil.copytree(ROOT / directory, target / directory)
    shutil.copy2(ROOT / "pyproject.toml", target / "pyproject.toml")
    return target


@pytest.mark.parametrize(
    ("relative_path", "list_key", "required"),
    [
        ("data/evaluation/project-stage-status-v1.json", "stages", STAGE_KEYS),
        ("data/evaluation/v1-release-gates.json", "gates", GATE_KEYS),
        ("data/evaluation/evaluation-evidence-index.json", "artifacts", EVIDENCE_KEYS),
        ("data/evaluation/known-issues-v1.json", "issues", ISSUE_KEYS),
        ("data/evaluation/v1-gap-closure-plan.json", "tasks", TASK_KEYS),
    ],
)
def test_stage12_artifact_schema(relative_path: str, list_key: str, required: set[str]) -> None:
    document = _json(ROOT / relative_path)
    assert document[list_key]
    assert all(required <= row.keys() for row in document[list_key])


def test_portfolio_base_gates_pass_and_v1_strict_fails() -> None:
    rc, rc_code = evaluate(ROOT, "rc", strict=True)
    v1, v1_code = evaluate(ROOT, "v1", strict=True)
    assert rc_code == 0
    assert rc["status"] == "passed"
    assert rc["recommended_rc_version"] == "v0.9.0-rc3"
    assert rc["highest_satisfied_version"] == "v1.0.0-portfolio"
    assert v1_code != 0
    assert v1["status"] == "failed"
    assert v1["highest_satisfied_version"] == "v1.0.0-portfolio"


def test_release_package_version_satisfies_rc_or_portfolio_gate() -> None:
    result, code = evaluate(ROOT, "rc", strict=True)
    gate = next(gate for gate in result["gates"] if gate["gate_id"] == "REL-01")
    assert code == 0
    assert gate["status"] == "passed"


def test_missing_artifact_fails(tmp_path: Path) -> None:
    root = _copy_workspace(tmp_path)
    (root / "data/evaluation/citation-human-audit-summary-v1.json").unlink()
    result, code = evaluate(root, "rc", strict=True)
    assert code != 0
    assert any(
        "citation-human-audit-summary-v1.json" in error
        for error in result["validation_errors"]
    )


def test_pending_gold_fails(tmp_path: Path) -> None:
    root = _copy_workspace(tmp_path)
    path = root / "data/evaluation/gold-set-v1.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["review_status"] = "pending"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    result, code = evaluate(root, "rc", strict=True)
    assert code != 0
    assert any(gate["gate_id"] == "DATA-05" for gate in result["unmet_gates"])


def test_enabled_reranker_fails_production_gate(tmp_path: Path) -> None:
    root = _copy_workspace(tmp_path)
    path = root / "src/paper_research/config.py"
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace("rerank_enabled: bool = False", "rerank_enabled: bool = True"),
        encoding="utf-8",
    )
    result, code = evaluate(root, "rc", strict=True)
    assert code != 0
    assert any(gate["gate_id"] == "RERANK-01" for gate in result["unmet_gates"])


def test_quality_and_deep_research_gates_block_v1() -> None:
    result, code = evaluate(ROOT, "v1", strict=True)
    blockers = {gate["gate_id"] for gate in result["unmet_gates"]}
    assert code != 0
    assert "QA-08" in blockers
    assert "DR-08" in blockers


def test_failed_attempts_cannot_be_hidden(tmp_path: Path) -> None:
    root = _copy_workspace(tmp_path)
    path = root / "data/evaluation/deep-research-smoke-v1.json"
    document = _json(path)
    document["failed_attempts"] = []
    path.write_text(json.dumps(document), encoding="utf-8")
    result, code = evaluate(root, "rc", strict=True)
    assert code != 0
    assert any("hides" in error for error in result["validation_errors"])


def test_oracle_metrics_cannot_be_production(tmp_path: Path) -> None:
    root = _copy_workspace(tmp_path)
    path = root / "data/evaluation/qa-context-diagnostics-v1.json"
    document = _json(path)
    oracle = next(row for row in document["runs"] if row.get("oracle"))
    oracle["production_metric"] = True
    path.write_text(json.dumps(document), encoding="utf-8")
    result, code = evaluate(root, "rc", strict=True)
    assert code != 0
    assert any("Oracle" in error for error in result["validation_errors"])


def test_output_does_not_include_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "stage12-test-secret-value"
    monkeypatch.setenv("LLM_API_KEY", secret)
    result, _ = evaluate(ROOT, "rc", strict=True)
    assert secret not in json.dumps(result)


def test_windows_compatible_path_with_spaces(tmp_path: Path) -> None:
    root = _copy_workspace(tmp_path)
    result, code = evaluate(root, "rc", strict=True)
    assert code == 0
    assert result["status"] == "passed"
