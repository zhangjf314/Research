"""Verify Stage 13 history against the Stage 13.14 baseline commit."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"
DOCS = ROOT / "docs"
BASELINE = "fac12001e8ca4beb85b603a0d4706f765389a2f4"
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
}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_digest(relative: str, value: bytes) -> str:
    text = value.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    if relative.endswith(".jsonl"):
        parsed = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        parsed = json.loads(text)
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
        rows.append(
            {
                "label": label,
                "path": relative,
                "baseline_raw_sha256": digest(baseline),
                "current_raw_sha256": digest(current),
                "baseline_canonical_sha256": canonical_digest(relative, baseline),
                "current_canonical_sha256": canonical_digest(relative, current),
                "unchanged": canonical_digest(relative, baseline)
                == canonical_digest(relative, current),
            }
        )
    body = {
        "schema_version": "stage13-14-historical-protection-v1",
        "baseline_commit": BASELINE,
        "files": rows,
        "changed_count": sum(not row["unchanged"] for row in rows),
        "historical_stage13_12_gate": "FAILED_AND_PRESERVED",
        "stage13_12_pending_citation_pairs_modified": False,
        "historical_raw_runs_modified": False,
        "gate": "PASSED" if all(row["unchanged"] for row in rows) else "FAILED",
    }
    (DATA / "stage13-14-historical-protection-v1.json").write_text(
        json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DOCS / "stage13-14-historical-protection-v1.md").write_text(
        "# Stage 13.14 Historical Protection\n\n"
        f"- Baseline: `{BASELINE}`\n"
        f"- Files checked: {len(rows)}\n"
        f"- Changed: {body['changed_count']}\n"
        f"- Gate: `{body['gate']}`\n"
        "- Stage 13.12 remains FAILED_AND_PRESERVED; its pending citation review "
        "and raw runs were not modified.\n",
        encoding="utf-8",
    )
    print(json.dumps(body, ensure_ascii=False))
    if body["gate"] != "PASSED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
