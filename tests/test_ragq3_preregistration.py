import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEZE = "artifacts/rag-quality-v3/a0/preregistration/pre-result-freeze-v1.json"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _is_ancestor(older: str, newer: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", older, newer], cwd=ROOT, check=False
        ).returncode
        == 0
    )


def test_ragq3_candidate_spec_is_preresult() -> None:
    """Candidate specification must exist in history before every result artifact."""
    freeze_commit = _git("log", "--diff-filter=A", "--format=%H", "--", FREEZE).splitlines()[0]
    freeze = json.loads((ROOT / FREEZE).read_text(encoding="utf-8"))
    assert freeze["status"] == "PRE_RESULT_FREEZE_COMMIT"
    assert freeze["candidate_results_before_freeze"] == 0

    result_paths = list((ROOT / "artifacts/rag-quality-v3/a1").glob("**/*.json"))
    for result_path in result_paths:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if "pre_result_freeze_commit" not in payload:
            continue
        assert payload["pre_result_freeze_commit"] == freeze_commit
        introduced = _git(
            "log", "--diff-filter=A", "--format=%H", "--", str(result_path.relative_to(ROOT))
        ).splitlines()[0]
        assert _is_ancestor(freeze_commit, introduced)


def test_ragq3_freeze_declares_representation_only_scope() -> None:
    matrix = json.loads(
        (ROOT / "artifacts/rag-quality-v3/a0/preregistration/candidate-matrix-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert [candidate["name"] for candidate in matrix["candidates"]] == [
        "Q3-R0",
        "Q3-R1",
        "Q3-R2",
        "Q3-R3",
        "Q3-R4",
    ]
    assert matrix["invariant_runtime_contract"]["reranker"] == "disabled"
