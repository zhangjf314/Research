"""Audit retrieval benchmark runtime modules for label leakage."""

from __future__ import annotations

import ast
import json

try:
    from scripts.stage13_27_common import DATA, DOCS, ROOT, write_json
except ModuleNotFoundError:
    from stage13_27_common import DATA, DOCS, ROOT, write_json

OUT_JSON = DATA / "retrieval-benchmark-feature-leakage-audit.json"
OUT_DOC = DOCS / "retrieval-benchmark-feature-leakage-audit.md"
SOURCES = [
    ROOT / "src" / "paper_research" / "retrieval" / "local_lexical_index.py",
    ROOT / "src" / "paper_research" / "retrieval" / "reciprocal_rank_fusion.py",
    ROOT / "src" / "paper_research" / "retrieval" / "obligation_query_builder_v1.py",
]

FORBIDDEN = {
    "gold_online_leakage": ("gold", "positive_core", "positive_equivalent"),
    "human_label_online_leakage": ("human", "reviewer", "review_status"),
    "oracle_label_online_leakage": ("oracle", "upper_bound"),
    "split_leakage": ("blind_holdout", "validation", "development"),
    "question_id_special_cases": ("question_id", "required_claim_id", "q001", "q002", "q050"),
    "fixed_block_special_cases": ("b000023", "1706.03762|2|b000023"),
}


def build() -> dict[str, object]:
    hits = {bucket: [] for bucket in FORBIDDEN}
    arg_hits: list[str] = []
    for path in SOURCES:
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        for bucket, needles in FORBIDDEN.items():
            for needle in needles:
                if needle in lowered:
                    hits[bucket].append(f"{path.relative_to(ROOT)}::{needle}")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.arg) and node.arg in {"question_id", "required_claim_id"}:
                arg_hits.append(f"{path.relative_to(ROOT)}::{node.arg}")
    body = {
        "schema_version": "retrieval-benchmark-feature-leakage-audit-v1",
        "sources": [str(path.relative_to(ROOT)) for path in SOURCES],
        **{bucket: len(items) for bucket, items in hits.items()},
        "string_hits": hits,
        "arg_hits": arg_hits,
    }
    body["gate"] = (
        "PASSED"
        if not any(body[bucket] for bucket in FORBIDDEN) and not arg_hits
        else "FAILED"
    )
    return body


def main() -> None:
    body = build()
    write_json(OUT_JSON, body)
    OUT_DOC.write_text(
        "# Retrieval Benchmark Feature Leakage Audit\n\n"
        f"- Gate: `{body['gate']}`\n"
        f"- Gold online leakage: `{body['gold_online_leakage']}`\n"
        f"- Human-label online leakage: `{body['human_label_online_leakage']}`\n"
        f"- Oracle-label online leakage: `{body['oracle_label_online_leakage']}`\n"
        f"- Split leakage: `{body['split_leakage']}`\n"
        f"- Question-ID special cases: `{body['question_id_special_cases']}`\n"
        f"- Fixed block special cases: `{body['fixed_block_special_cases']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(body, indent=2))


if __name__ == "__main__":
    main()
