"""Audit evidence-selection-v2 candidate for prohibited online features."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "evaluation"
DOCS = ROOT / "docs"
SOURCE = ROOT / "src" / "paper_research" / "generation" / "evidence_selection_v2.py"
OUT_JSON = DATA / "evidence-selection-v2-feature-leakage-audit.json"
OUT_DOC = DOCS / "evidence-selection-v2-feature-leakage-audit.md"

PROHIBITED = {
    "claim-evidence-gold-dev-v1",
    "gold-set-v1",
    "retrieval-gold-v2",
    "core_gold",
    "equivalent_valid_evidence",
    "human_label",
    "human_labels",
    "question_id",
    "required_claim_id",
    "failure_taxonomy",
}


def audit_source() -> dict[str, Any]:
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    string_hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            lower = node.value.lower()
            for token in PROHIBITED:
                if token in lower:
                    string_hits.append(node.value)
    arg_hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.arg in {"question_id", "required_claim_id"}:
            arg_hits.append(node.arg)
    body = {
        "schema_version": "evidence-selection-v2-feature-leakage-audit-v1",
        "source": str(SOURCE.relative_to(ROOT)),
        "gold_online_leakage": 0 if not string_hits else len(string_hits),
        "human_label_online_leakage": 0
        if not any("human" in hit.lower() for hit in string_hits)
        else 1,
        "fixed_id_special_cases": len(arg_hits),
        "string_hits": string_hits,
        "arg_hits": arg_hits,
        "gate": "PASSED" if not string_hits and not arg_hits else "FAILED",
        "allowed_context": [
            "evaluation scorer",
            "offline replay",
            "tests",
        ],
    }
    return body


def main() -> None:
    body = audit_source()
    OUT_JSON.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_DOC.write_text(
        "# Evidence Selection v2 Feature Leakage Audit\n\n"
        f"- Source: `{body['source']}`\n"
        f"- Gold online leakage: `{body['gold_online_leakage']}`\n"
        f"- Human-label online leakage: `{body['human_label_online_leakage']}`\n"
        f"- Fixed-ID special cases: `{body['fixed_id_special_cases']}`\n"
        f"- Gate: `{body['gate']}`\n\n"
        "The selector is allowed to be used in offline replay and tests. Production selection "
        "must not read Gold relations, human labels, fixed question IDs, fixed claim IDs, "
        "or failure taxonomy labels.\n",
        encoding="utf-8",
    )
    if body["gate"] != "PASSED":
        raise RuntimeError("EVIDENCE_SELECTION_V2_FEATURE_LEAKAGE_DETECTED")
    print(json.dumps(body, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
