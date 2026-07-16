"""Backfill deterministic raw-payload validation artifacts without changing outcomes."""

from __future__ import annotations

import json

from paper_research.generation.schema_reliability import MinimalRequiredClaimsPayload

try:
    from scripts.evidence_qa_dev_v3_3_lib import RUN_ROOT
except ModuleNotFoundError:
    from evidence_qa_dev_v3_3_lib import RUN_ROOT  # type: ignore[no-redef]

try:
    from scripts.evidence_qa_dev_lib_v1 import canonical_hash
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import canonical_hash  # type: ignore[no-redef]


def main() -> None:
    updated = []
    for result_path in sorted(RUN_ROOT.glob("live-dev-v3-3-*/final-result.json")):
        run_dir = result_path.parent
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["status"] != "validation_failed":
            continue
        response = json.loads(
            (run_dir / "raw-provider-response.json").read_text(encoding="utf-8")
        )
        raw = json.loads(response["choices"][0]["message"]["content"])
        structural = MinimalRequiredClaimsPayload.model_validate(raw).model_dump(mode="json")
        (run_dir / "raw-model-payload.json").write_text(
            json.dumps(structural, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (run_dir / "payload-validation.json").write_text(
            json.dumps(
                {
                    "json_valid": True,
                    "structural_schema_valid": True,
                    "slot_cardinality_valid": len(structural["required_claim_results"])
                    == result["required_claim_count"],
                    "answerability_protocol_valid": False,
                    "failure_type": result["failure_type"],
                    "failure_reason": result["failure_reason"],
                    "payload_hash": canonical_hash(structural),
                    "normalization_used": False,
                    "outcome_modified": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        updated.append(result["run_id"])
    print(json.dumps({"finalized_validation_artifacts": updated}, ensure_ascii=False))


if __name__ == "__main__":
    main()
