"""Research-only ATOMIC_TEXT_PROTOCOL_V2; no production defaults or V1 changes."""

from pydantic import Field, create_model, field_validator

from paper_research.wiki.synthesis_bundle import candidate_scope
from paper_research.wiki_synthesis import Strict
from paper_research.wiki_synthesis_r1 import R1Statement, R1Synthesis, SourceUnit, validate_r1

PROTOCOL = "ATOMIC_TEXT_PROTOCOL_V2"
MAX_TEXT = 600


def atomic_text_v2(value):
    if not isinstance(value, str):
        raise ValueError("TEXT_NOT_STRING")
    if not value.strip():
        raise ValueError("TEXT_EMPTY_OR_WHITESPACE")
    if len(value) > MAX_TEXT:
        raise ValueError("TEXT_OVER_MAX_LENGTH")
    for char in value:
        code = ord(char)
        if code < 0x20 or 0x7F <= code <= 0x9F or code in (0x2028, 0x2029):
            raise ValueError("TEXT_CONTROL_OR_LINE_SEPARATOR")
        if 0xD800 <= code <= 0xDFFF:
            raise ValueError("TEXT_UNPAIRED_SURROGATE")
    return value


class _StatementV2(Strict):
    text: str = Field(min_length=1, max_length=MAX_TEXT)
    source_units: list[SourceUnit] = Field(min_length=1)
    evidence_span_ids: list[str] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def structural_text(cls, value):
        return atomic_text_v2(value)

    @field_validator("source_units")
    @classmethod
    def original_units(cls, units):
        return R1Statement.reject_duplicate_units(units)

    @field_validator("evidence_span_ids")
    @classmethod
    def original_evidence(cls, values):
        return R1Statement.unchanged_evidence_id_syntax(values)


# Preserve the public JSON schema's exact names, fields, limits and $refs.
StatementV2 = create_model("R1Statement", __base__=_StatementV2)
SynthesisV2 = create_model(
    "R1Synthesis",
    __base__=Strict,
    statements=(list[StatementV2], Field(min_length=1, max_length=6)),
)


def validate_v2(value, bundle):
    validated = SynthesisV2.model_validate(value)
    # Fully validated V2 fields enter the original immutable R1 identity/locality/
    # coverage validator. Its existing-instance path skips only V1 text revalidation.
    # There is no text replacement, model patch, or relaxed ID/coverage check.
    bridge = R1Synthesis.model_construct(statements=validated.statements)
    checked = validate_r1(bridge, candidate_scope(bundle))
    if checked is not bridge:
        raise ValueError("ORIGINAL_VALIDATOR_BRIDGE_REVALIDATION_DRIFT")
    return validated


def derived_text(text):
    """CommonMark numeric entities are lossless data, never Markdown structure."""
    atomic_text_v2(text)
    return "".join(f"&#{ord(char)};" for char in text)


def render_v2(output, bundle):
    lines = ["# " + derived_text(bundle["question"]["question_text"]), ""]
    ids = sorted({sid for statement in output.statements for sid in statement.evidence_span_ids})
    numbered = {sid: index + 1 for index, sid in enumerate(ids)}
    for statement in output.statements:
        lines.append(
            "- "
            + derived_text(statement.text)
            + " "
            + " ".join(f"[^e{numbered[sid]}]" for sid in statement.evidence_span_ids)
        )
    lines.append("")
    for sid in ids:
        lines.append(f"[^e{numbered[sid]}]: Evidence " + derived_text(sid))
    return "\n".join(lines) + "\n"
