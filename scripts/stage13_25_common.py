"""Shared helpers for Stage 13.25 offline set-completion audits."""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
from typing import Any

from paper_research.generation.citation_selection import CitationCandidate

try:
    common = importlib.import_module("scripts.stage13_23_common")
except (ImportError, ModuleNotFoundError):
    common = importlib.import_module("stage13_23_common")

DATA = common.DATA
DOCS = common.DOCS
ROOT = common.ROOT
RUN_ROOT = common.RUN_ROOT
candidate_rows = common.candidate_rows
canonical_hash = common.canonical_hash
file_hash = common.file_hash
final_slot = common.final_slot
load_gold = common.load_gold
read_json = common.read_json
registry_maps = common.registry_maps
relation_sets = common.relation_sets
selected_runs = common.selected_runs
write_json = common.write_json


def to_candidate(row: dict[str, Any]) -> CitationCandidate:
    return CitationCandidate(
        citation_id=row["citation_id"],
        paper_id=row["paper_id"],
        page=row["page"],
        block_id=row["block_id"],
        text=row["text"],
        neighboring_context=row.get("neighboring_context", ""),
        evidence_role=tuple(row.get("evidence_role", [])),
        retrieval_origin=row.get("retrieval_origin", "original_selected"),
        original_selected=bool(row.get("original_selected", False)),
        adjacent_completion=bool(row.get("adjacent_completion", False)),
        currently_cited=bool(row.get("currently_cited", False)),
        retrieval_score=float(row.get("retrieval_score", 0.0)),
        lexical_alignment=float(row.get("lexical_alignment", 0.0)),
        numeric_coverage=float(row.get("numeric_coverage", 0.0)),
        comparison_side_coverage=float(row.get("comparison_side_coverage", 0.0)),
        claim_scope_coverage=float(row.get("claim_scope_coverage", 0.0)),
        redundancy_group=row.get("redundancy_group"),
        token_cost=int(row.get("token_cost", 0)),
    )


def citation_keys(
    citation_ids: list[str] | tuple[str, ...],
    key_by_citation: dict[str, str],
) -> set[str]:
    return {key_by_citation[cid] for cid in citation_ids if cid in key_by_citation}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def iter_claim_contexts() -> list[dict[str, Any]]:
    runs = selected_runs()
    contexts: list[dict[str, Any]] = []
    for required_claim_id, record in sorted(load_gold().items()):
        question_id = record["question_id"]
        run_dir = RUN_ROOT / runs[question_id]
        _registry, key_by_citation = registry_maps(run_dir)
        final = final_slot(run_dir, required_claim_id)
        candidates = tuple(to_candidate(row) for row in candidate_rows(run_dir, required_claim_id))
        contexts.append(
            {
                "question_id": question_id,
                "required_claim_id": required_claim_id,
                "record": record,
                "run_dir": run_dir,
                "key_by_citation": key_by_citation,
                "registry_keys": set(key_by_citation.values()),
                "final": final,
                "claim_text": final.get("claim_text") or record["required_claim_text"],
                "baseline_ids": tuple(final["citation_ids"]),
                "candidates": candidates,
                "candidate_ids": {candidate.citation_id for candidate in candidates},
                "valid_sets": relation_sets(record),
            }
        )
    return contexts
