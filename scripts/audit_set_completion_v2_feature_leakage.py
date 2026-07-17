"""Audit set-completion v2 production modules for forbidden offline features."""

from __future__ import annotations

import ast
import json
import re

try:
    from scripts.stage13_25_common import DATA, DOCS, ROOT, write_json
except ModuleNotFoundError:
    from stage13_25_common import DATA, DOCS, ROOT, write_json  # type: ignore[no-redef]

SOURCES = [
    ROOT / "src" / "paper_research" / "generation" / "claim_obligations.py",
    ROOT / "src" / "paper_research" / "generation" / "set_completion_v2.py",
]
OUT_JSON = DATA / "set-completion-v2-feature-leakage-audit.json"
OUT_DOC = DOCS / "set-completion-v2-feature-leakage-audit.md"

PATTERNS = {
    "gold_online_leakage": (r"\bgold\b", r"core_gold", r"equivalent_valid"),
    "human_label_online_leakage": (r"\bhuman\b", r"reviewer", r"human_label"),
    "oracle_label_online_leakage": (r"\boracle\b",),
    "attribution_label_online_leakage": (r"root_cause", r"failure_attribution", r"gap_cause"),
    "fixed_id_special_cases": (r"\bq\d{3}\b", r"cl-q\d{3}", r"\b\d{4}\.\d{4,5}\b", r"b\d{6}"),
}


def build() -> dict:
    hits = {key: [] for key in PATTERNS}
    arg_hits = []
    for source in SOURCES:
        text = source.read_text(encoding="utf-8")
        for key, patterns in PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.I):
                    hits[key].append(f"{source.name}:{pattern}")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.arg) and node.arg in {
                "question_id",
                "required_claim_id",
                "gold",
                "human_label",
                "oracle",
                "gap_cause",
            }:
                arg_hits.append(f"{source.name}:{node.arg}")
    body = {
        "schema_version": "set-completion-v2-feature-leakage-audit-v1",
        "sources": [str(source.relative_to(ROOT)) for source in SOURCES],
        **{key: len(value) for key, value in hits.items()},
        "string_hits": hits,
        "arg_hits": sorted(set(arg_hits)),
    }
    body["fixed_id_special_cases"] += len(body["arg_hits"])
    body["gate"] = (
        "PASSED"
        if all(body[key] == 0 for key in PATTERNS)
        else "FAILED"
    )
    return body


def main() -> None:
    body = build()
    write_json(OUT_JSON, body)
    OUT_DOC.write_text(
        "# Set Completion v2 Feature Leakage Audit\n\n"
        f"- Gate: `{body['gate']}`\n"
        f"- Gold online leakage: `{body['gold_online_leakage']}`\n"
        f"- Human-label online leakage: `{body['human_label_online_leakage']}`\n"
        f"- Oracle-label online leakage: `{body['oracle_label_online_leakage']}`\n"
        f"- Attribution-label online leakage: `{body['attribution_label_online_leakage']}`\n"
        f"- Fixed-ID special cases: `{body['fixed_id_special_cases']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(body, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
