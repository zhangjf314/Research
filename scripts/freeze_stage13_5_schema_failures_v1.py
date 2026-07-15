# ruff: noqa: E501,E702,F841
"""Freeze immutable Stage 13.5 schema-failure evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DOCS, canonical_hash
    from scripts.evidence_qa_dev_v3_lib import RUN_ROOT
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DATA, DOCS, canonical_hash  # type: ignore[no-redef]
    from evidence_qa_dev_v3_lib import RUN_ROOT  # type: ignore[no-redef]

OUTPUT = DATA / "stage13-5-schema-failure-freeze-v1.json"
OUTPUT_DOC = DOCS / "stage13-5-schema-failure-freeze-v1.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_content(run_dir: Path) -> str:
    body = json.loads((run_dir / "raw-provider-response.json").read_text(encoding="utf-8"))
    return body["choices"][0]["message"]["content"]


def freeze_record(run_dir: Path) -> dict[str, Any]:
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "run-metadata.json").read_text(encoding="utf-8"))
    envelope = json.loads((run_dir / "provider-response-envelope.json").read_text(encoding="utf-8"))
    expected = json.loads((run_dir / "required-claims-input.json").read_text(encoding="utf-8"))
    try:
        raw = json.loads(raw_content(run_dir)); valid = True
    except json.JSONDecodeError:
        raw, valid = None, False
    text = raw_content(run_dir)
    keys = list(raw) if isinstance(raw, dict) else []
    observed_claim_ids = sorted(set(expected_claim["required_claim_id"] for expected_claim in expected["required_claims"]) & set(_walk_keys(raw)))
    return {"question_id": result["question_id"], "run_id": result["run_id"], "raw_response_hash": sha(run_dir / "raw-provider-response.json"), "provider_response_envelope_hash": sha(run_dir / "provider-response-envelope.json"), "request_id": metadata["request_id"], "usage": result["usage"], "registry_hash": metadata["citation_registry_hash"], "required_claim_input_hash": metadata["required_claim_input_hash"], "validation_error": result["failure_reason"], "raw_json_valid": valid, "response_top_level_type": type(raw).__name__ if raw is not None else "malformed", "response_top_level_keys": keys, "wrapper_shape": keys[0] if len(keys) == 1 and isinstance(raw.get(keys[0]), dict) else None, "contains_all_required_claim_ids": len(observed_claim_ids) == len(expected["required_claims"]), "contains_citation_ids": bool(__import__("re").findall(r'\bE\d{3}\b', text)), "contains_free_triple": any(key in {"paper_id", "page", "block_id"} for key in _walk_keys(raw)), "official_status": result["status"]}


def _walk_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [str(key) for key in value] + [item for child in value.values() for item in _walk_keys(child)]
    if isinstance(value, list):
        return [item for child in value for item in _walk_keys(child)]
    return []


def main() -> None:
    run_dirs = sorted(path.parent for path in RUN_ROOT.glob("live-dev-v3-*/result.json"))
    records = sorted((freeze_record(path) for path in run_dirs), key=lambda row: row["question_id"])
    if len(records) != 10 or any(row["official_status"] != "validation_failed" for row in records):
        raise RuntimeError("Stage 13.5 frozen run set is not exactly ten validation failures")
    signature = canonical_hash(records)
    payload = {"schema_version": "stage13-5-schema-failure-freeze-v1", "freeze_signature": signature, "record_count": 10, "official_schema_success": 0, "official_required_claim_coverage": {"numerator": 0, "denominator": 27}, "records": records, "historical_results_modified": False}
    if OUTPUT.exists() and json.loads(OUTPUT.read_text(encoding="utf-8")).get("freeze_signature") != signature:
        raise RuntimeError("existing freeze signature differs")
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_DOC.write_text(f"# Stage 13.5 Schema Failure Freeze\n\n- Frozen runs: 10\n- Freeze signature: `{signature}`\n- Official schema success: 0/10\n- Official required-claim coverage: 0/27\n- Historical runs and metrics were not modified.\n", encoding="utf-8")
    print(json.dumps({"records": 10, "freeze_signature": signature}))


if __name__ == "__main__":
    main()
