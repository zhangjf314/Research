from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EXPECTED_STAGE2_FINAL_CONFIG_HASH = (
    "995a144385180b2931ec2c6366f7f7306301a42d77ad7c85f4be9e6d9e5091d9"
)


class RAGBackendLockError(RuntimeError):
    """Raised when the Stage 3 frozen RAG backend lock drifts."""


def validate_rag_backend_lock(
    path: Path = Path("data/evaluation/research-agent/stage3-rag-backend-lock-v1.json"),
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    backend = payload.get("rag_backend") or {}
    expected = {
        "retrieval": "Current Hybrid",
        "reranker": "disabled",
        "query_rewrite": "disabled",
        "query_decomposition": "disabled",
        "context_selector": "baseline",
    }
    if backend != expected:
        raise RAGBackendLockError("RAG_BACKEND_LOCK_MISMATCH")
    if payload.get("stage2_final_config_hash") != EXPECTED_STAGE2_FINAL_CONFIG_HASH:
        raise RAGBackendLockError("RAG_BACKEND_LOCK_MISMATCH")
    return payload

