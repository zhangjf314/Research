from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_full_test_suite_sharded.ps1"


def _workspace_tmp(name: str) -> Path:
    path = ROOT / ".runtime" / "pytest-sharded-script-tests" / f"{name}-{uuid.uuid4().hex}"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _run_sharded(
    work_dir: Path,
    tests_dir: Path,
    *extra_args: str,
) -> subprocess.CompletedProcess[str]:
    out_dir = work_dir / "shards"
    command = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(SCRIPT),
        "-PythonPath",
        sys.executable,
        "-TestsPath",
        str(tests_dir),
        "-OutputDirectory",
        str(out_dir),
        *extra_args,
    ]
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=120,
    )


def _write_test(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_sharded_runner_fails_but_continues_after_failed_shard() -> None:
    work_dir = _workspace_tmp("failure")
    try:
        tests_dir = work_dir / "tests"
        _write_test(tests_dir / "test_a.py", "def test_pass():\n    assert True\n")
        _write_test(tests_dir / "test_b.py", "def test_fail():\n    assert False\n")
        _write_test(
            tests_dir / "test_c.py",
            "def test_after_failure_still_runs():\n    assert True\n",
        )

        result = _run_sharded(work_dir, tests_dir)

        assert result.returncode == 1, result.stdout + result.stderr
        summary = json.loads((work_dir / "shards" / "summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "FAILED"
        assert summary["collected_total"] == 3
        assert summary["executed_total"] == 3
        assert summary["failed"] == 1
        assert summary["passed_shards"] == 2
        assert summary["failed_shards"] == 1
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_sharded_runner_resume_does_not_rerun_successful_shards() -> None:
    work_dir = _workspace_tmp("resume")
    try:
        tests_dir = work_dir / "tests"
        marker = work_dir / "marker.txt"
        marker_literal = str(marker).replace("\\", "\\\\")
        _write_test(
            tests_dir / "test_marker.py",
            f"from pathlib import Path\n"
            f"def test_marker_once():\n"
            f"    marker = Path(r'{marker_literal}')\n"
            f"    previous = int(marker.read_text()) if marker.exists() else 0\n"
            f"    marker.write_text(str(previous + 1))\n"
            f"    assert True\n",
        )

        first = _run_sharded(work_dir, tests_dir)
        second = _run_sharded(work_dir, tests_dir, "-Resume")

        assert first.returncode == 0
        assert second.returncode == 0
        assert marker.read_text(encoding="utf-8") == "1"
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_sharded_runner_sums_junit_counts() -> None:
    work_dir = _workspace_tmp("counts")
    try:
        tests_dir = work_dir / "tests"
        _write_test(
            tests_dir / "test_counts.py",
            "import pytest\n"
            "def test_pass():\n    assert True\n"
            "def test_skip():\n    pytest.skip('intentional')\n",
        )

        result = _run_sharded(work_dir, tests_dir)

        assert result.returncode == 0
        summary = json.loads((work_dir / "shards" / "summary.json").read_text(encoding="utf-8"))
        assert summary["collected_total"] == 2
        assert summary["executed_total"] == 2
        assert summary["passed"] == 1
        assert summary["skipped"] == 1
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_sharded_runner_script_contains_missing_file_gate() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "missing_test_files" in text
    assert "--lf" not in text
    assert "--ff" not in text
