"""Gold-free canonical runtime provenance for paper ingestion and indexing.

This module deliberately has no dependency on evaluation datasets, questions, or
Gold lineage.  It is evaluation infrastructure only until separately promoted.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


def _stable(namespace: str, value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{namespace}:{hashlib.sha256(payload.encode()).hexdigest()}"


@dataclass(frozen=True)
class NeutralSourceBlock:
    parser_local_id: str
    text: str
    section_path: tuple[str, ...]
    page_start: int
    page_end: int


@dataclass(frozen=True)
class CanonicalRuntimeUnit:
    canonical_unit_id: str
    canonical_document_id: str
    neutral_source_block_ids: tuple[str, ...]
    source_spans: tuple[tuple[int, int], ...]
    section_path: tuple[str, ...]
    text: str
    text_sha256: str

    def index_payload(self) -> dict[str, object]:
        """The complete permitted runtime payload; intentionally Gold-free."""
        return {
            "unit_id": self.canonical_unit_id,
            "canonical_document_id": self.canonical_document_id,
            "neutral_source_block_ids": list(self.neutral_source_block_ids),
            "source_spans": [list(item) for item in self.source_spans],
            "section_path": list(self.section_path),
            "text_sha256": self.text_sha256,
        }


class CanonicalPaperIngestor:
    """Build deterministic source-derived structural units without Gold inputs."""

    representation_version = "gold-free-structural-v1"

    def canonical_document_id(self, source_sha256: str) -> str:
        return _stable("canonical_document", {"source_sha256": source_sha256})

    def neutral_block_id(self, document_id: str, parser_local_id: str) -> str:
        return _stable(
            "neutral_source_block",
            {"canonical_document_id": document_id, "parser_local_id": parser_local_id},
        )

    def build_units(
        self, *, source_sha256: str, blocks: list[NeutralSourceBlock]
    ) -> list[CanonicalRuntimeUnit]:
        document_id = self.canonical_document_id(source_sha256)
        units: list[CanonicalRuntimeUnit] = []
        for block in blocks:
            text = " ".join(block.text.split())
            if not text:
                continue
            neutral_id = self.neutral_block_id(document_id, block.parser_local_id)
            unit_id = _stable(
                "canonical_runtime_unit",
                {
                    "representation_version": self.representation_version,
                    "canonical_document_id": document_id,
                    "neutral_source_block_ids": [neutral_id],
                    "text": text,
                    "section_path": list(block.section_path),
                    "source_span": [block.page_start, block.page_end],
                },
            )
            units.append(
                CanonicalRuntimeUnit(
                    canonical_unit_id=unit_id,
                    canonical_document_id=document_id,
                    neutral_source_block_ids=(neutral_id,),
                    source_spans=((block.page_start, block.page_end),),
                    section_path=block.section_path,
                    text=text,
                    text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                )
            )
        return units
