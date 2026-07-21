"""Audit Evidence Selection v3 production module for prohibited features."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "evaluation"
DOCS = ROOT / "docs"
SOURCE = ROOT / "src" / "paper_research" / "generation" / "evidence_selection_v3.py"
OUT = DATA / "evidence-selection-v3-feature-leakage-audit.json"
DOC = DOCS / "evidence-selection-v3-feature-leakage-audit.md"

PROHIBITED = {
    "claim-evidence-gold-dev-v1",
    "retrieval-gold-v2",
    "core_gold",
    "equivalent_valid",
    "human_label",
    "failure_attribution",
    "primary_root_cause",
    "question_id",
    "required_claim_id",
    "relation_key",
}


def audit() -> dict:
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    string_hits: list[str] = []
    arg_hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.lower()
            for token in PROHIBITED:
                if token in value:
                    string_hits.append(node.value)
        if isinstance(node, ast.arg) and node.arg in {"question_id", "required_claim_id"}:
            arg_hits.append(node.arg)
    body = {
        "schema_version": "evidence-selection-v3-feature-leakage-audit-v1",
        "source": str(SOURCE.relative_to(ROOT)),
        "gold_online_leakage": 0 if not string_hits else len(string_hits),
        "human_label_online_leakage": 0
        if not any("human" in hit.lower() for hit in string_hits)
        else 1,
        "attribution_label_online_leakage": 0
        if not any("root_cause" in hit.lower() or "failure" in hit.lower() for hit in string_hits)
        else 1,
        "fixed_id_special_cases": len(arg_hits),
        "string_hits": string_hits,
        "arg_hits": arg_hits,
        "gate": "PASSED" if not string_hits and not arg_hits else "FAILED",
    }
    return body


def main() -> None:
    body = audit()
    OUT.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    DOC.write_text(
        "# Evidence Selection v3 Feature Leakage Audit\n\n"
        f"- Gate: `{body['gate']}`\n"
        f"- Gold online leakage: `{body['gold_online_leakage']}`\n"
        f"- Human-label online leakage: `{body['human_label_online_leakage']}`\n"
        f"- Attribution-label online leakage: `{body['attribution_label_online_leakage']}`\n"
        f"- Fixed-ID special cases: `{body['fixed_id_special_cases']}`\n",
        encoding="utf-8",
    )
    if body["gate"] != "PASSED":
        raise RuntimeError("EVIDENCE_SELECTION_V3_FEATURE_LEAKAGE_DETECTED")
    print(json.dumps(body, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
