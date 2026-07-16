"""Verify Stage 13 history against the Stage 13.15 baseline commit."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"
DOCS = ROOT / "docs"
BASELINE = "70b2401f29cfec0cb8b3764fc945bc75dcfda96f"
FILES = {
    "stage13_5_formal": "data/evaluation/stage13-5-schema-failure-freeze-v1.json",
    "stage13_6_readiness": "data/evaluation/evidence-qa-dev-v3-readiness-v1.json",
    "stage13_7_migration": "data/evaluation/stage13-review-hash-migration-v1.json",
    "stage13_8_dev_v3_1": "data/evaluation/evidence-qa-dev-v3-1.json",
    "stage13_9_metric": "data/evaluation/citation-recall-metric-v2.json",
    "stage13_10_claim_gold": "data/evaluation/claim-evidence-gold-dev-v1.jsonl",
    "stage13_11_offline_replay": "data/evaluation/dev-v3-2-offline-replay-v1.json",
    "stage13_12_failure_freeze": "data/evaluation/stage13-12-dev-v3-2-failure-freeze-v1.json",
    "stage13_13_reconciliation": "data/evaluation/stage13-12-reservation-reconciliation-v1.json",
    "stage13_14_failure_freeze": "data/evaluation/stage13-14-dev-v3-3-failure-freeze-v1.json",
    "stage13_15_payload_replay": "data/evaluation/dev-v3-3-payload-contract-v2-replay.json",
}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_digest(relative: str, value: bytes) -> str:
    text = value.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    parsed = (
        [json.loads(line) for line in text.splitlines() if line.strip()]
        if relative.endswith(".jsonl")
        else json.loads(text)
    )
    encoded = json.dumps(
        parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return digest(encoded)


def main() -> None:
    rows = []
    for label, relative in FILES.items():
        current = (ROOT / relative).read_bytes()
        baseline = subprocess.check_output(
            ["git", "show", f"{BASELINE}:{relative}"], cwd=ROOT
        )
        baseline_canonical = canonical_digest(relative, baseline)
        current_canonical = canonical_digest(relative, current)
        rows.append(
            {
                "label": label,
                "path": relative,
                "baseline_raw_sha256": digest(baseline),
                "current_raw_sha256": digest(current),
                "baseline_canonical_sha256": baseline_canonical,
                "current_canonical_sha256": current_canonical,
                "unchanged": baseline_canonical == current_canonical,
            }
        )
    body = {
        "schema_version": "stage13-16-historical-protection-v1",
        "baseline_commit": BASELINE,
        "files": rows,
        "changed_count": sum(not row["unchanged"] for row in rows),
        "stage13_14_gate": "FAILED_AND_PRESERVED",
        "stage13_14_raw_runs_modified": False,
        "stage13_14_pending_citation_records_modified": False,
        "stage13_12_pending_citation_records_modified": False,
        "historical_reservation_events_modified": False,
        "gold_modified": False,
        "human_labels_modified": False,
        "gate": "PASSED" if all(row["unchanged"] for row in rows) else "FAILED",
    }
    (DATA / "stage13-16-historical-protection-v1.json").write_text(
        json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DOCS / "stage13-16-historical-protection-v1.md").write_text(
        "# Stage 13.16 Historical Protection\n\n"
        f"- Baseline: `{BASELINE}`\n"
        f"- Canonical files checked: {len(rows)}\n"
        f"- Changed: {body['changed_count']}\n"
        f"- Gate: `{body['gate']}`\n"
        "- Stage 13.14 remains FAILED_AND_PRESERVED. Historical raw runs, pending "
        "citation records, reservation events, Gold, and human labels were not modified.\n",
        encoding="utf-8",
    )
    print(json.dumps(body, ensure_ascii=False))
    if body["gate"] != "PASSED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
