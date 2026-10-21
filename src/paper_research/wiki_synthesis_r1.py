"""Offline R1 lineage contract. SourceUnit is an anchor, never evidence or truth."""
# ruff: noqa: E501 -- preserve the frozen prompt's exact lines

from typing import Literal

from pydantic import Field, field_validator, model_validator

from paper_research.wiki_synthesis import (
    GENERATOR_PROMPT,
    Statement,
    Strict,
    judge_input,
)

PROTOCOL_ID = "WIKI_SYNTHESIS_SOURCE_UNIT_V1"
OLD_SOURCE_BLOCK = """Visible source claims are linked_pages with type ClaimPage; use their source_node_id
as source_claim_ids. Every visible claim ID must appear at least once. Each statement
must explicitly list all its source claim IDs and all supporting evidence span IDs
from bundle.evidence. Cite at least one visible evidence span per statement. Multiple
claims may be combined only into one independently judgeable factual assertion."""
NEW_SOURCE_BLOCK = """Use source_units selected by code from the current bundle: all visible ClaimPages
if any exist; otherwise all visible RequirementPages. Use their source_node_id as
id with type CLAIM or REQUIREMENT, respectively. Do not choose or mix fallback types.
Every code-selected synthesis unit must appear at least once in source_units. Each
statement must explicitly list all its source units and all supporting evidence span
IDs from bundle.evidence. Cite at least one visible evidence span per statement.
Source_units identify which frozen semantic unit is being synthesized. They do not
themselves justify the factual statement. Every factual statement must be supported
by the cited evidence spans. Requirement text is an information need, not evidence;
do not answer from requirement text alone. Multiple source units may be combined
only into one independently judgeable factual assertion."""
if GENERATOR_PROMPT.count(OLD_SOURCE_BLOCK) != 1:
    raise ValueError("OLD_SOURCE_PROMPT_BLOCK_DRIFT")
R1_GENERATOR_PROMPT = GENERATOR_PROMPT.replace(OLD_SOURCE_BLOCK, NEW_SOURCE_BLOCK)


class SourceUnit(Strict):
    """Semantic lineage anchor; its ID/type is not a support judgment."""

    type: Literal["CLAIM", "REQUIREMENT"]
    id: str = Field(min_length=1)

    @model_validator(mode="after")
    def node_type_matches(self):
        prefix = {"CLAIM": "claim:", "REQUIREMENT": "requirement:"}[self.type]
        if not self.id.startswith(prefix) or len(self.id) == len(prefix):
            raise ValueError("SOURCE_UNIT_NODE_TYPE_MISMATCH")
        return self


class R1Statement(Strict):
    text: str = Field(min_length=1, max_length=600)
    source_units: list[SourceUnit] = Field(min_length=1)
    evidence_span_ids: list[str] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def unchanged_atomic_syntax(cls, value):
        return Statement.atomic_syntax(value)

    @field_validator("evidence_span_ids")
    @classmethod
    def unchanged_evidence_id_syntax(cls, value):
        return Statement.unique_ids(value)

    @field_validator("source_units")
    @classmethod
    def reject_duplicate_units(cls, units):
        pairs = [(u.type, u.id) for u in units]
        if len(pairs) != len(set(pairs)):
            raise ValueError("DUPLICATE_SOURCE_UNIT")
        return units


class R1Synthesis(Strict):
    statements: list[R1Statement] = Field(min_length=1, max_length=6)


def select_source_units(bundle):
    """Code owns the mode. Zero claims is the sole requirement fallback trigger."""
    claims = [p for p in bundle["linked_pages"] if p["type"] == "ClaimPage"]
    requirements = [p for p in bundle["linked_pages"] if p["type"] == "RequirementPage"]
    pages, kind = (claims, "CLAIM") if claims else (requirements, "REQUIREMENT")
    if not pages:
        raise ValueError("NO_ELIGIBLE_VISIBLE_SOURCE_UNIT")
    units = [SourceUnit(type=kind, id=p["source_node_id"]) for p in pages]
    if len({u.id for u in units}) != len(units):
        raise ValueError("DUPLICATE_VISIBLE_SOURCE_UNIT")
    return sorted(units, key=lambda u: (u.type, u.id))


def validate_r1(value, bundle):
    output = R1Synthesis.model_validate(value)
    selected = {(u.type, u.id) for u in select_source_units(bundle)}
    visible_evidence = {e["span_id"] for e in bundle["evidence"]}
    seen = set()
    for statement in output.statements:
        references = {(u.type, u.id) for u in statement.source_units}
        if not references <= selected:
            raise ValueError("SOURCE_UNIT_NOT_ELIGIBLE_IN_CURRENT_BUNDLE")
        if not set(statement.evidence_span_ids) <= visible_evidence:
            raise ValueError("EVIDENCE_NOT_VISIBLE_IN_CURRENT_BUNDLE")
        seen.update(references)
    if seen != selected:
        raise ValueError("MISSING_SELECTED_SOURCE_UNIT_REFERENCE")
    return output


def source_unit_coverage(output, bundle):
    validate_r1(output.model_dump(), bundle)
    # Exact union equality above is structural/provenance coverage, not semantic recall.
    return 1.0


def evidence_relationships(units, bundle, graph):
    """Offline visibility audit, using frozen graph edges, not semantic support labels."""
    visible = {e["span_id"] for e in bundle["evidence"]}
    records = []
    for unit in units:
        edges = [
            e
            for e in graph.edges.values()
            if e.type == "GROUNDED_BY" and e.source == unit.id and e.target in visible
        ]
        records.append(
            {
                "source_unit": unit.model_dump(),
                "accessible_evidence_span_ids": sorted({e.target for e in edges}),
                "relationship_edge_ids": sorted(e.id for e in edges),
                "has_visible_evidence_relationship": bool(edges),
            }
        )
    return records


def structural_witness(bundle, units, relationships):
    """Synthetic syntax witness only; never a factual synthesis candidate."""
    if not units or not all(r["has_visible_evidence_relationship"] for r in relationships):
        raise ValueError("SOURCE_UNIT_OR_VISIBLE_EVIDENCE_MISSING")
    spans = sorted({sid for r in relationships for sid in r["accessible_evidence_span_ids"]})
    value = {
        "statements": [
            {
                "text": "Offline structural witness.",
                "source_units": [u.model_dump() for u in units],
                "evidence_span_ids": spans,
            }
        ]
    }
    return validate_r1(value, bundle)


def r1_judge_input(statement, bundle):
    # Reuse the unchanged one-statement / cited-text implementation. It never reads units.
    return judge_input(statement, bundle)
