"""Focused compatibility checks for product admission and persistent task locks."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from paper_research.agents.research_agent.checkpoint import JsonResearchAgentCheckpointStore
from paper_research.indexing.identity import validate_trust_record
from paper_research.indexing.registry import IndexRegistry


@pytest.mark.parametrize("kind", ["task", "corpus"])
def test_os_lease_excludes_another_process_and_releases(tmp_path, kind):
    task = JsonResearchAgentCheckpointStore(tmp_path / "checkpoints")
    registry = IndexRegistry(tmp_path / "index_registry.json")
    code = """
import sys
from pathlib import Path
from paper_research.agents.research_agent.checkpoint import JsonResearchAgentCheckpointStore
from paper_research.indexing.registry import IndexRegistry
root, kind = Path(sys.argv[1]), sys.argv[2]
lease = (JsonResearchAgentCheckpointStore(root/'checkpoints').task_write_lease('lease-test')
         if kind == 'task' else IndexRegistry(root/'index_registry.json').lease('query'))
try:
    with lease:
        print('ACQUIRED')
except Exception as exc:
    print(exc.code)
"""
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
           "PYTHONDONTWRITEBYTECODE": "1"}

    def child():
        result = subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path), kind],
                                env=env, capture_output=True, text=True, timeout=30, check=True)
        return result.stdout.strip().splitlines()[-1]

    lease = task.task_write_lease("lease-test") if kind == "task" else registry.lease("query")
    with lease:
        assert child() == ("RESEARCH_TASK_BUSY" if kind == "task" else "CORPUS_INTEGRITY_BUSY")
    assert child() == "ACQUIRED"


def test_operator_scope_is_explicit_and_synthetic_scope_remains_valid():
    intent = {"schema_version": "corpus-admission-intent-v1",
              "admission_id": "00000000-0000-4000-8000-000000000001",
              "authorization_ref": "local-approval", "authority_id": "operator",
              "sources": [{"source_ref": "source-0", "pdf_sha256": "a" * 64,
                           "byte_count": 20, "input_relative_path": "inputs/input.pdf"}],
              "created_at": "2026-10-15T00:00:00+00:00", "intent_digest": "0" * 64}
    for scope in ("EXPLICIT_USER_AUTHORIZED_SYNTHETIC_PDFS", "EXPLICIT_OPERATOR_AUTHORIZED_PDFS"):
        validate_trust_record("admission_intent", {**intent, "closed_input_scope": scope})
    with pytest.raises(ValueError):
        validate_trust_record("admission_intent", {**intent, "closed_input_scope": "PUBLIC_UPLOAD"})


def test_cli_requires_explicit_approval(monkeypatch):
    from paper_research.admission import main

    monkeypatch.setattr(sys, "argv", ["admission", "admit", "--authorization-ref", "reason", "x.pdf"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
