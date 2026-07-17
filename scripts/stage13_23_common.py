"""Shared helpers for Stage 13.23 offline audits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "evaluation"
DOCS = ROOT / "docs"
RUN_ROOT = DATA / "evidence-qa-dev-v3-6" / "runs"


def canonical_hash(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def selected_runs() -> dict[str, str]:
    summary = read_json(DATA / "evidence-qa-dev-v3-6.json")
    return {
        row["question_id"]: row["run_id"]
        for row in summary["attempt_history"]
        if row["selected"]
    }


def relation_key(row: dict[str, Any]) -> str:
    return f"{row['paper_id']}|{row['page']}|{row['block_id']}"


def load_gold() -> dict[str, dict[str, Any]]:
    return {
        row["required_claim_id"]: row
        for row in read_jsonl(DATA / "claim-evidence-gold-dev-v1.jsonl")
        if row["answerable"]
    }


def relation_sets(record: dict[str, Any]) -> dict[str, set[str]]:
    by_id = {rel["relation_id"]: rel for rel in record["candidate_evidence_relations"]}
    core_ids = set(record.get("approved_core_relations", []))
    supporting_ids = set(record.get("approved_supporting_relations", []))
    equivalent_ids = set(record.get("equivalent_non_gold_relations", []))
    return {
        "core": {relation_key(by_id[rel_id]) for rel_id in core_ids if rel_id in by_id},
        "supporting": {
            relation_key(by_id[rel_id]) for rel_id in supporting_ids if rel_id in by_id
        },
        "equivalent": {
            relation_key(by_id[rel_id]) for rel_id in equivalent_ids if rel_id in by_id
        },
    }


def registry_maps(run_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    registry = read_json(run_dir / "citation-registry.json")
    by_id = {entry["citation_id"]: entry for entry in registry["entries"]}
    key_by_id = {
        entry["citation_id"]: f"{entry['paper_id']}|{entry['page']}|{entry['block_id']}"
        for entry in registry["entries"]
    }
    return by_id, key_by_id


def candidate_rows(run_dir: Path, required_claim_id: str) -> list[dict[str, Any]]:
    local = read_json(run_dir / "candidate-evidence-local.json")
    for row in local["candidate_rows"]:
        if row["required_claim_id"] == required_claim_id:
            return list(row["candidates"])
    return []


def final_slot(run_dir: Path, required_claim_id: str) -> dict[str, Any]:
    result = read_json(run_dir / "final-result.json")
    for row in result["final_answer"]["required_claim_results"]:
        if row["required_claim_id"] == required_claim_id:
            return row
    raise KeyError(required_claim_id)
