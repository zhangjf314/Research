"""Audit Evidence Selection v4 production module for forbidden offline features."""

from __future__ import annotations

import ast
import json
import re

try:
    from scripts.stage13_23_common import DATA, DOCS, ROOT, write_json
except ModuleNotFoundError:
    from stage13_23_common import DATA, DOCS, ROOT, write_json  # type: ignore[no-redef]

SOURCE = ROOT / "src" / "paper_research" / "generation" / "evidence_selection_v4.py"
OUT_JSON = DATA / "evidence-selection-v4-feature-leakage-audit.json"
OUT_DOC = DOCS / "evidence-selection-v4-feature-leakage-audit.md"

GOLD_PATTERNS = (r"\bgold\b", r"core_gold", r"equivalent_valid", r"retrieval_gold")
HUMAN_PATTERNS = (r"\bhuman\b", r"reviewer", r"reviewed_at", r"human_label")
ATTRIBUTION_PATTERNS = (r"root_cause", r"failure_attribution", r"wrong_evidence_class")
FIXED_ID_PATTERNS = (
    r"\bq\d{3}\b",
    r"cl-q\d{3}",
    r"\b\d{4}\.\d{4,5}\b",
    r"\|b\d{6}\b",
)


def string_hits(text: str, patterns: tuple[str, ...]) -> list[str]:
    hits: list[str] = []
    for pattern in patterns:
        if re.search(pattern, text, re.I):
            hits.append(pattern)
    return hits


def arg_hits(tree: ast.AST) -> list[str]:
    forbidden = {
        "question_id",
        "required_claim_id",
        "gold",
        "human_label",
        "root_cause",
        "fixed_relation_key",
    }
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.arg in forbidden:
            hits.append(node.arg)
        if isinstance(node, ast.Name) and node.id in forbidden:
            hits.append(node.id)
    return sorted(set(hits))


def build() -> dict:
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    gold_hits = string_hits(text, GOLD_PATTERNS)
    human_hits = string_hits(text, HUMAN_PATTERNS)
    attribution_hits = string_hits(text, ATTRIBUTION_PATTERNS)
    fixed_hits = string_hits(text, FIXED_ID_PATTERNS)
    args = arg_hits(tree)
    body = {
        "schema_version": "evidence-selection-v4-feature-leakage-audit-v1",
        "source": str(SOURCE.relative_to(ROOT)),
        "gold_online_leakage": len(gold_hits),
        "human_label_online_leakage": len(human_hits),
        "attribution_label_online_leakage": len(attribution_hits),
        "fixed_id_special_cases": len(fixed_hits) + len(args),
        "string_hits": {
            "gold": gold_hits,
            "human": human_hits,
            "attribution": attribution_hits,
            "fixed_ids": fixed_hits,
        },
        "arg_hits": args,
    }
    body["gate"] = (
        "PASSED"
        if body["gold_online_leakage"] == 0
        and body["human_label_online_leakage"] == 0
        and body["attribution_label_online_leakage"] == 0
        and body["fixed_id_special_cases"] == 0
        else "FAILED"
    )
    return body


def write_outputs(body: dict) -> None:
    write_json(OUT_JSON, body)
    OUT_DOC.write_text(
        "# Evidence Selection v4 Feature Leakage Audit\n\n"
        f"- Gate: `{body['gate']}`\n"
        f"- Gold online leakage: `{body['gold_online_leakage']}`\n"
        f"- Human-label online leakage: `{body['human_label_online_leakage']}`\n"
        f"- Attribution-label online leakage: `{body['attribution_label_online_leakage']}`\n"
        f"- Fixed-ID special cases: `{body['fixed_id_special_cases']}`\n",
        encoding="utf-8",
    )


def main() -> None:
    body = build()
    write_outputs(body)
    print(json.dumps(body, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
