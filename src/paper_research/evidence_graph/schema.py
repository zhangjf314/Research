"""Strict canonical records and collision-resistant, unambiguous identities."""

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

NodeType = Literal["Paper", "Chunk", "EvidenceSpan", "Requirement", "Claim", "Question"]
EdgeType = Literal["CONTAINS", "SOURCED_FROM", "GROUNDED_BY", "REQUIRES"]
EdgeKind = Literal["structural", "evidence-semantic"]


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def node_id(kind: NodeType, *parts: str | int) -> str:
    if not parts or any(type(p) not in (str, int) or p == "" for p in parts):
        raise ValueError("INVALID_ID_PARTS")
    return kind.lower() + ":" + digest([kind, list(parts)])


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Provenance(StrictRecord):
    source_phase: str = Field(min_length=1)
    source_artifact: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_record_id: str = Field(min_length=1)
    construction_method: Literal["deterministic_identity", "frozen_gold_mapping"]
    evidence_status: Literal["STRUCTURAL_SOURCE", "FROZEN_MODEL_GOLD_NOT_HUMAN_TRUTH"]
    paper_id: str | None = None
    chunk_id: str | None = None
    span_id: str | None = None


class Node(StrictRecord):
    id: str = Field(min_length=1)
    type: NodeType
    properties: dict[str, Any]
    provenance: Provenance


class Edge(StrictRecord):
    id: str = Field(min_length=1)
    type: EdgeType
    kind: EdgeKind
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    properties: dict[str, Any]
    provenance: Provenance

    @model_validator(mode="after")
    def category_and_truth(self):
        semantic = self.type == "GROUNDED_BY"
        if semantic != (self.kind == "evidence-semantic"):
            raise ValueError("EDGE_CATEGORY_MISMATCH")
        if semantic and (
            self.provenance.construction_method != "frozen_gold_mapping"
            or self.provenance.evidence_status != "FROZEN_MODEL_GOLD_NOT_HUMAN_TRUTH"
            or not all(
                (self.provenance.paper_id, self.provenance.chunk_id, self.provenance.span_id)
            )
        ):
            raise ValueError("SEMANTIC_PROVENANCE_REQUIRED")
        if not semantic and self.provenance.construction_method != "deterministic_identity":
            raise ValueError("STRUCTURAL_PROVENANCE_REQUIRED")
        return self


def edge_id(edge: Edge | dict) -> str:
    value = edge.model_dump() if isinstance(edge, Edge) else dict(edge)
    value.pop("id", None)
    return "edge:" + digest(value)


class GraphManifest(StrictRecord):
    schema_version: Literal["EVIDENCE_GRAPH_V1"]
    benchmark: str
    source_hashes: dict[str, str]
    span_policy: Literal["FULL_CANONICAL_CHUNK_HALF_OPEN_UNICODE_CODEPOINTS"]
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    nodes_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    edges_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    scientific_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifier_predictions_as_truth: Literal[False]
