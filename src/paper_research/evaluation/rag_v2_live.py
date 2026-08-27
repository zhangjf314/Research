"""RAG v2 RQ9 live-dense tooling: provenance, index identity, embedding-text
builders with citation invariance, collection isolation, paired metrics.

No provider calls live in this module — it defines contracts and pure helpers
so they are unit-testable offline.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

LIVE_DENSE_SCHEMA_VERSION = "rag-v2-live-dense-v1"
MIN_DENSE_CANDIDATES = 20

# Experiment collections must use one of these prefixes and must never
# collide with production/eval collection names (RQ9/RQ11 isolation).
EXPERIMENT_COLLECTION_PREFIXES = (
    "papers_jina_ragv2_rq9_",
    "papers_jina_ragv2_rq11_holdout_",
    "papers_jina_ragv2_rq15_holdout_v2_",
)
EXPERIMENT_COLLECTION_PREFIX = "papers_jina_ragv2_rq9_"  # legacy constant

PROTECTED_COLLECTIONS = (
    "papers_jina_eval34",
    "papers_hash_eval34",
    "papers_hash_v1",
    "papers_jina_v5_text_small_1024__jina_embeddings_v5_text_small_v1",
    "paper_chunks",
)


class DenseProvenance(StrEnum):
    """Every dense ranking must declare which leg produced it (RQ9 §1)."""

    PROXY_DENSE = "PROXY_DENSE"  # dev-v3.6 heuristic candidate replay
    PRODUCTION_JINA_DENSE = "PRODUCTION_JINA_DENSE"  # live Jina query + Qdrant


def collection_isolation_guard(name: str) -> None:
    """Raise unless `name` is a legal, isolated experiment collection."""

    if not any(
        name.startswith(prefix)
        for prefix in EXPERIMENT_COLLECTION_PREFIXES
    ):
        raise ValueError(
            f"experiment collection must start with one of "
            f"{EXPERIMENT_COLLECTION_PREFIXES}: {name}"
        )
    for fragment in PROTECTED_COLLECTIONS:
        if name == fragment:
            raise ValueError(f"refusing to touch protected collection: {name}")


def embedding_config_hash(
    *,
    provider: str,
    model: str,
    dimensions: int,
    distance: str,
    query_task: str,
    passage_task: str,
    chunk_max_tokens: int,
    chunk_overlap_tokens: int,
    representation: str,
) -> str:
    """Deterministic identity for an embedding/index configuration."""

    payload = json.dumps(
        {
            "provider": provider,
            "model": model,
            "dimensions": dimensions,
            "distance": distance,
            "query_task": query_task,
            "passage_task": passage_task,
            "chunk_max_tokens": chunk_max_tokens,
            "chunk_overlap_tokens": chunk_overlap_tokens,
            "representation": representation,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def derive_paper_title(blocks: Sequence[dict[str, Any]]) -> str:
    """Title block if present, else first heading, else empty (recorded)."""

    for block in blocks:
        if block.get("block_type") == "title":
            return (block.get("text") or "").strip()[:300]
    for block in blocks:
        if block.get("block_type") == "heading":
            return (block.get("text") or "").strip()[:300]
    return ""


def build_embedding_text(
    representation: str,
    chunk_text: str,
    *,
    paper_title: str = "",
    section_path: Sequence[str] = (),
) -> tuple[str, str]:
    """Return (embedding_text, citation_text) for E0/E1/E2 (RQ9 §8).

    citation_text is ALWAYS the original chunk text — enrichment may change
    what gets embedded but never what gets cited (citation invariance).
    """

    if representation == "E0":
        return chunk_text, chunk_text
    if representation == "E1":
        parts = [paper_title, " / ".join(section_path), chunk_text]
        return "\n".join(p for p in parts if p), chunk_text
    if representation == "E2":
        headings = list(section_path[:2])  # section + subsection
        parts = [paper_title, *headings, chunk_text]
        return "\n".join(p for p in parts if p), chunk_text
    raise KeyError(f"unknown representation: {representation}")


def paired_comparison(
    dense_rank: int, bm25_rank: int, *, pool_depth: int = 20
) -> str:
    """Paired per-claim verdict; misses count as rank pool_depth+1."""

    d = dense_rank if dense_rank else pool_depth + 1
    b = bm25_rank if bm25_rank else pool_depth + 1
    if d < b:
        return "dense_wins"
    if b < d:
        return "bm25_wins"
    return "ties"


def dense_relation(
    dense_rank: int, bm25_rank: int, hybrid_rank: int, *, pool_depth: int = 20
) -> str:
    """DENSE_HELPED / HURT / NEUTRAL for production dense vs BM25 (RQ9 §6)."""

    d = dense_rank if dense_rank else pool_depth + 1
    b = bm25_rank if bm25_rank else pool_depth + 1
    h = hybrid_rank if hybrid_rank else pool_depth + 1
    if d < b:
        return "DENSE_HELPED"
    if h > b:
        return "DENSE_HURT"
    return "DENSE_NEUTRAL"


@dataclass
class LiveDenseTrace:
    """Per-claim live trace record (RQ9 §4 capture contract)."""

    query_id: str
    claim_id: str
    query_text: str
    query_type: str
    gold_block_ids: list[str]
    dense_candidates: list[dict[str, Any]]  # block_id/rank/score/paper/page/section
    bm25_ranking: list[str]
    equal_rrf_ranking: list[str]
    final_context_ranking: list[str]
    provenance: DenseProvenance
    provenance_note: str = ""
    index_identity: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": LIVE_DENSE_SCHEMA_VERSION,
            "query_id": self.query_id,
            "claim_id": self.claim_id,
            "query_text": self.query_text,
            "query_type": self.query_type,
            "gold_block_ids": self.gold_block_ids,
            "dense_provenance": self.provenance.value,
            "provenance_note": self.provenance_note,
            "index_identity": self.index_identity,
            "dense_candidates": self.dense_candidates,
            "dense_depth": len(self.dense_candidates),
            "bm25_ranking": self.bm25_ranking,
            "equal_rrf_ranking": self.equal_rrf_ranking,
            "final_context_ranking": self.final_context_ranking,
        }

    def validate(self) -> list[str]:
        problems = []
        if self.provenance == DenseProvenance.PROXY_DENSE:
            problems.append("PROXY_DENSE traces are not valid live evidence")
        if len(self.dense_candidates) < MIN_DENSE_CANDIDATES:
            problems.append(
                "LIVE_DENSE_DEPTH_INSUFFICIENT: "
                f"{len(self.dense_candidates)} < {MIN_DENSE_CANDIDATES}"
            )
        return problems
