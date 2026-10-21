"""Strict page identities, exact records and snapshot-local provenance."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from paper_research.evidence_graph.schema import digest

VERSION = "AGENT_NATIVE_WIKI_FOUNDATION_V1"
PAGE_TYPES = {
    "Paper": "PaperPage",
    "Question": "QuestionPage",
    "Requirement": "RequirementPage",
    "Claim": "ClaimPage",
}
DEFAULT_BUDGET = {"max_evidence_blocks": 4, "max_linked_pages": 8, "max_chars": 24000}


def page_id(node_type: str, node_identity: str) -> str:
    if node_type not in PAGE_TYPES or not node_identity.startswith(node_type.lower() + ":"):
        raise ValueError("PAGE_SOURCE_TYPE_MISMATCH")
    return "wiki:" + node_type.lower() + ":" + node_identity


def uri(identity: str) -> str:
    _, kind, source = identity.split(":", 2)
    if identity != page_id(kind.title(), source):
        raise ValueError("INVALID_WIKI_PAGE_ID")
    return "wiki://" + kind + "/" + source


def filename(identity: str) -> str:
    return digest(identity)


class Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class Footer(Strict):
    graph_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_node_id: str = Field(min_length=1)
    source_edge_ids: list[str]
    source_artifact_hashes: dict[str, str]
    wiki_builder_version: Literal["AGENT_NATIVE_WIKI_FOUNDATION_V1"]
    dependency_digest: str = Field(pattern=r"^[a-f0-9]{64}$")


class Page(Strict):
    id: str
    type: Literal["PaperPage", "QuestionPage", "RequirementPage", "ClaimPage"]
    source_node_id: str
    title: str
    source_record: dict[str, Any]
    links: dict[str, list[str]]
    backlinks: list[str]
    linked_records: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    provenance_paths: list[list[str]]
    evidence_status: Literal["FROZEN_MODEL_GOLD_NOT_HUMAN_TRUTH"]
    provenance: Footer
