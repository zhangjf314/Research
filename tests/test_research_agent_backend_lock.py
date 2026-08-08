import json

import pytest

from paper_research.agents.research_agent.backend_lock import (
    RAGBackendLockError,
    validate_rag_backend_lock,
)


def test_backend_lock_accepts_frozen_stage2_hash() -> None:
    payload = validate_rag_backend_lock()
    assert payload["rag_backend"]["retrieval"] == "Current Hybrid"
    assert payload["rag_backend"]["reranker"] == "disabled"


def test_backend_lock_rejects_drift(tmp_path) -> None:
    path = tmp_path / "lock.json"
    path.write_text(
        json.dumps(
            {
                "rag_backend": {
                    "retrieval": "Current Hybrid",
                    "reranker": "enabled",
                    "query_rewrite": "disabled",
                    "query_decomposition": "disabled",
                    "context_selector": "baseline",
                },
                "stage2_final_config_hash": "bad",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RAGBackendLockError):
        validate_rag_backend_lock(path)

