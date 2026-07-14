"""Build deterministic EvidenceUnit artifacts from existing parsed Production blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from paper_research.evidence.schema import EvidenceUnit, build_evidence_unit
from paper_research.parsing.types import PaperBlock

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/evaluation/production-corpus-v1.json"
OUTPUT = ROOT / "data/evaluation/evidence-corpus-v1.jsonl"
MANIFEST = ROOT / "data/evaluation/evidence-corpus-v1-manifest.json"
REPORT = ROOT / "docs/evidence-corpus-v1.md"
SOURCE_VERSION = "production-corpus-v1:parsed-blocks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--report", type=Path, default=REPORT)
    return parser.parse_args()


def _docker_read(database_id: str, filename: str) -> str:
    container_path = f"/app/data/parsed/{database_id}/{filename}"
    command = ["docker", "compose", "exec", "-T", "api", "cat", container_path]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"cannot read existing parsed artifact {container_path}: {message}")
    return result.stdout.decode("utf-8")


def read_source(database_id: str, filename: str, source_root: Path | None) -> str:
    if source_root is not None:
        path = source_root / database_id / filename
        if not path.is_file():
            raise RuntimeError(f"missing parsed source: {path}")
        return path.read_text(encoding="utf-8")
    local = ROOT / "data/parsed" / database_id / filename
    if local.is_file():
        return local.read_text(encoding="utf-8")
    return _docker_read(database_id, filename)


def _records(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _chunk_map(chunks: list[dict]) -> dict[str, str]:
    mapping: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        for block_id in chunk.get("block_ids", []):
            mapping[block_id].append(chunk["chunk_id"])
    return {block_id: sorted(ids)[0] for block_id, ids in mapping.items()}


def _signature(units: list[EvidenceUnit]) -> str:
    canonical = "\n".join(
        json.dumps(unit.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        for unit in units
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def build(source_root: Path | None = None) -> tuple[list[EvidenceUnit], dict]:
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    included = [paper for paper in corpus["papers"] if paper["included_in_production"]]
    if len(included) != 34:
        raise RuntimeError(f"expected 34 Production documents, got {len(included)}")
    units: list[EvidenceUnit] = []
    paper_stats = []
    for paper in sorted(included, key=lambda item: item["paper_id"]):
        block_rows = _records(read_source(paper["database_id"], "paper_blocks.jsonl", source_root))
        chunk_rows = _records(read_source(paper["database_id"], "paper_chunks.jsonl", source_root))
        chunks_by_block = _chunk_map(chunk_rows)
        paper_units = [
            build_evidence_unit(
                paper["paper_id"],
                PaperBlock.model_validate(row),
                source_chunk_id=chunks_by_block.get(row["block_id"]),
                source_version=SOURCE_VERSION,
            )
            for row in block_rows
        ]
        units.extend(paper_units)
        roles = Counter(role for unit in paper_units for role in unit.evidence_roles)
        types = Counter(unit.block_type for unit in paper_units)
        non_evidence = sum(not unit.eligible_for_final_context for unit in paper_units)
        paper_stats.append(
            {
                "paper_id": paper["paper_id"],
                "database_id": paper["database_id"],
                "block_count": len(block_rows),
                "evidence_unit_count": len(paper_units),
                "eligible_evidence_count": len(paper_units) - non_evidence,
                "non_evidence_count": non_evidence,
                "block_type_distribution": dict(sorted(types.items())),
                "evidence_role_distribution": dict(sorted(roles.items())),
                "unclassified_count": sum(not unit.evidence_roles for unit in paper_units),
                "missing_page_or_block_id_count": sum(
                    not unit.block_id or unit.page < 1 for unit in paper_units
                ),
                "short_unit_count": sum(len(unit.normalized_text) < 20 for unit in paper_units),
                "long_unit_count": sum(len(unit.normalized_text) > 4000 for unit in paper_units),
            }
        )
    units.sort(key=lambda unit: (unit.paper_id, unit.ordinal, unit.block_id))
    if len({unit.evidence_id for unit in units}) != len(units):
        raise RuntimeError("duplicate deterministic evidence_id")
    manifest = {
        "schema_version": "evidence-corpus-manifest-v1",
        "source_version": SOURCE_VERSION,
        "source_corpus": "data/evaluation/production-corpus-v1.json",
        "production_document_count": len(included),
        "excluded_ocr_fixture_count": corpus["excluded_ocr_fixtures"],
        "evidence_unit_count": len(units),
        "eligible_evidence_count": sum(unit.eligible_for_final_context for unit in units),
        "non_evidence_count": sum(not unit.eligible_for_final_context for unit in units),
        "build_signature": _signature(units),
        "papers": paper_stats,
    }
    manifest_hash_input = json.dumps(manifest, sort_keys=True, ensure_ascii=False)
    manifest["manifest_hash"] = hashlib.sha256(manifest_hash_input.encode()).hexdigest()
    return units, manifest


def write(units: list[EvidenceUnit], manifest: dict, output: Path, report: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(unit.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
            for unit in units
        ),
        encoding="utf-8",
    )
    rows = manifest["papers"]
    lines = [
        "# Evidence Corpus v1",
        "",
        "Derived deterministically from existing parsed blocks for the signed 34-document "
        "Production corpus. No PDF was reparsed and no Qdrant collection was modified.",
        "",
        f"- Evidence units: {manifest['evidence_unit_count']}",
        f"- Eligible units: {manifest['eligible_evidence_count']}",
        f"- Metadata/citation/non-evidence units: {manifest['non_evidence_count']}",
        f"- Build signature: `{manifest['build_signature']}`",
        f"- Manifest hash: `{manifest['manifest_hash']}`",
        "",
        "| Paper | Blocks | Eligible | Non-evidence | Short | Long | Missing IDs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['paper_id']} | {row['block_count']} | "
            f"{row['eligible_evidence_count']} | {row['non_evidence_count']} | "
            f"{row['short_unit_count']} | {row['long_unit_count']} | "
            f"{row['missing_page_or_block_id_count']} |"
        )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    units, manifest = build(args.source_root)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write(units, manifest, args.output, args.report)
    print(
        json.dumps(
            {
                "documents": manifest["production_document_count"],
                "evidence_units": manifest["evidence_unit_count"],
                "build_signature": manifest["build_signature"],
            }
        )
    )


if __name__ == "__main__":
    main()
