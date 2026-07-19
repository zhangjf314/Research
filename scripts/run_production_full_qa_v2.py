"""Stage 13.34 Production Full QA v2 runner.

The v2 runner reuses the v1 Docker API batch implementation but writes to
separate v2 artifacts and records lineage/config metadata. It does not mix v1
and v2 rows.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import scripts.run_production_full_qa_v1 as v1

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "evaluation"
DOCS = ROOT / "docs"
ARTIFACTS = ROOT / "artifacts"

v1.FULL_QA_JSON = DATA / "full-qa-production-v2.json"
v1.FULL_QA_CSV = DATA / "full-qa-production-v2.csv"
v1.FULL_QA_ITEMS = DATA / "full-qa-production-items-v2.jsonl"
v1.FULL_QA_TRACE = ARTIFACTS / "full-qa-production-trace-v2.json"
v1.FULL_QA_AUDIT_DOC = DOCS / "full-qa-production-audit-v2.md"
v1.FULL_QA_SUMMARY_DOC = DOCS / "full-qa-production-summary-v2.md"


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _postprocess() -> None:
    if not v1.FULL_QA_JSON.exists():
        return
    payload: dict[str, Any] = json.loads(v1.FULL_QA_JSON.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or {}
    completed = int(metrics.get("completed") or 0)
    failed = int(metrics.get("failed") or 0)
    unsupported = int(metrics.get("unsupported_claim_count") or 0)
    answerable_completed = int(metrics.get("answerable_items_completed") or 0)
    unsupported_rate = (
        round(unsupported / max(1, answerable_completed * 3), 6)
        if answerable_completed
        else None
    )
    quality_pass = (
        completed == 50
        and failed == 0
        and metrics.get("citation_id_validity") == 1.0
        and (metrics.get("required_claim_coverage") or 0) >= 0.70
        and (metrics.get("citation_precision") or 0) >= 0.70
        and (metrics.get("citation_recall") or 0) >= 0.65
        and (unsupported_rate or 1) <= 0.10
    )
    payload.update(
        {
            "schema_version": "full-qa-production-v2",
            "run_id": f"full-qa-production-v2-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
            "config_hash": _safe_sha256(
                {
                    "prompt_version": "qa-production-v1",
                    "normalizer_version": "citation-key-contract-v2",
                    "context_selector_version": "intent-aware-token-budget-v2",
                    "reranker_enabled": False,
                    "transport_retry_count": 1,
                }
            ),
            "dataset_hash": _path_sha256(v1.RETRIEVAL_GOLD),
            "normalizer_version": "citation-key-contract-v2",
            "context_selector_version": "intent-aware-token-budget-v2",
            "transport_retry_count": 1,
            "qa_generation_retry_count": 0,
            "json_repair_count": 0,
            "citation_repair_count": 0,
            "unsupported_claim_rate": unsupported_rate,
            "production_full_qa_gate": (
                "PASSED" if quality_pass else "FAILED_GROUNDING_QUALITY"
            )
            if failed == 0
            else "COMPLETED_WITH_FAILURES",
            "ready_for_production_deep_research": bool(quality_pass),
            "bound_git_commit": _git_head(),
        }
    )
    v1.write_json(v1.FULL_QA_JSON, payload)
    lineage = [
        "# Full QA Result Lineage v1",
        "",
        "- v1 status: `COMPLETED_WITH_FAILURES`",
        "- v2 status: `" + payload["status"] + "`",
        "- v1 and v2 are not mixed; v2 is a complete rerun with separate artifacts.",
        "- v2 changes: citation-key output contract, fail-closed schema normalization, "
        "compact JSON prompt, transport-only retry, and v2 context grounding audit.",
        f"- v2 bound git commit: `{payload['bound_git_commit']}`",
        "",
        "This is a 50-item human-reviewed internal development evaluation, not a blind benchmark.",
    ]
    (DOCS / "full-qa-result-lineage-v1.md").write_text(
        "\n".join(lineage) + "\n", encoding="utf-8"
    )


def _safe_sha256(payload: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _path_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    code = v1.main()
    _postprocess()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
