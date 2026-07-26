import uuid
from types import SimpleNamespace

from scripts.cleanup_audit_fixture_papers_v1 import classify


def test_fixture_classification_requires_audit_id_or_deterministic_fixture_name() -> None:
    fixture_id = uuid.uuid4()
    fixture = SimpleNamespace(
        id=fixture_id,
        title="fully-scanned-20260720152244",
        source_type="upload",
    )
    classification, reason = classify(fixture, set())
    assert classification == "CONFIRMED_AUDIT_FIXTURE"
    assert "fixture pattern" in reason


def test_non_fixture_is_not_auto_deleted() -> None:
    paper = SimpleNamespace(
        id=uuid.uuid4(),
        title="A real paper about scanned historical archives",
        source_type="upload",
    )
    classification, _ = classify(paper, set())
    assert classification == "REAL_USER_PAPER"

