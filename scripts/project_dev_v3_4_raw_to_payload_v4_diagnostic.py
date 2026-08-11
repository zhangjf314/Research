"""Project Stage 13.16 raw payloads to v4 without mutating history."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from scripts.evidence_qa_dev_lib_v1 import DEV_IDS
    from scripts.payload_contract_v4_lib import RUN_ROOT, project_raw_payload_v4
    from scripts.replay_dev_v3_4_payload_contract_v3 import historical_raw_payload
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DEV_IDS  # type: ignore[no-redef]
    from payload_contract_v4_lib import (  # type: ignore[no-redef]
        RUN_ROOT,
        project_raw_payload_v4,
    )
    from replay_dev_v3_4_payload_contract_v3 import historical_raw_payload  # type: ignore[no-redef]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_projection_rows() -> list[dict[str, Any]]:
    rows = []
    for question_id in DEV_IDS:
        run_dirs = sorted(RUN_ROOT.glob(f"live-dev-v3-4-{question_id}-*"))
        if run_dirs:
            run_dir = run_dirs[0]
            run_id = run_dir.name
            raw = load_json(run_dir / "raw-model-payload.json")
        else:
            run_id, raw = historical_raw_payload(question_id)
        projection = project_raw_payload_v4(raw)
        rows.append(
            {
                "question_id": question_id,
                "run_id": run_id,
                **projection,
            }
        )
    return rows


def main() -> None:
    rows = build_projection_rows()
    print(
        json.dumps(
            {
                "schema_version": "dev-v3-4-payload-v4-diagnostic-projection-v1",
                "questions": len(rows),
                "field_projection_completed": sum(
                    row["field_projection_completed"] for row in rows
                ),
                "semantic_conflicts": sum(
                    row.get("semantic_conflict", False) for row in rows
                ),
                "placeholder_fields_removed": sum(
                    row.get("placeholder_fields_removed", 0) for row in rows
                ),
                "semantic_field_modifications": sum(
                    row["semantic_field_modifications"] for row in rows
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
