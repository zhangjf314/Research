"""Static leakage audit separating production selection from offline evaluation."""

from __future__ import annotations

import json
import re

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DOCS
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DATA, DOCS  # type: ignore[no-redef]

ROOT = DATA.parents[1]
OUTPUT = DATA / "dev-v3-2-feature-leakage-audit-v1.json"
DOC = DOCS / "dev-v3-2-feature-leakage-audit-v1.md"
PRODUCTION_FILES = [
    ROOT / "src/paper_research/generation/citation_selection.py",
]
FORBIDDEN = [
    "claim-evidence-gold-dev-v1",
    "gold-set-v1",
    "retrieval-gold-v2",
    "human_label",
    "fully_supported",
    "partially_supported",
    "core_gold",
    "equivalent_valid_evidence",
]


def main() -> None:
    findings = []
    fixed_ids = []
    for path in PRODUCTION_FILES:
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN:
            if token.lower() in text.lower():
                findings.append({"path": str(path.relative_to(ROOT)), "token": token})
        fixed_ids.extend(
            {"path": str(path.relative_to(ROOT)), "value": value}
            for value in re.findall(r"\bq\d{3}\b|\bcl-q\d{3}-[a-f0-9]+\b", text)
        )
    payload = {
        "schema_version": "dev-v3-2-feature-leakage-audit-v1",
        "production_files_scanned": [str(path.relative_to(ROOT)) for path in PRODUCTION_FILES],
        "forbidden_feature_findings": findings,
        "fixed_id_findings": fixed_ids,
        "gold_leakage": bool(findings),
        "human_label_leakage": any("human" in item["token"] for item in findings),
        "fixed_id_special_cases": bool(fixed_ids),
        "evaluation_path_may_read_gold": True,
        "production_path_may_read_gold": False,
        "gate": "PASSED" if not findings and not fixed_ids else "FAILED",
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    DOC.write_text(
        "# Dev v3.2 feature leakage audit v1\n\n"
        f"- Production files scanned: {len(PRODUCTION_FILES)}\n"
        f"- Gold/human-label findings: {len(findings)}\n"
        f"- Fixed-ID findings: {len(fixed_ids)}\n"
        f"- Gate: **{payload['gate']}**\n\n"
        "Evaluation replay may read frozen Gold and human labels only after selection for scoring. "
        "The production selection path may not read them.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    if payload["gate"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
