"""Opt-in V1_3. Frozen V1_2 scope/task plus span binding and exact singleton wrapping."""

import copy
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field, StrictStr, create_model, model_validator

from paper_research.agents.research_agent.compact_verifier import (
    RATIONALE_NOT_REQUESTED,
    assessment_input,
)
from paper_research.agents.research_agent.contract_verifier import StructuredContractAdapter
from paper_research.agents.research_agent.evidence_contract import (
    EvidenceAssessment,
    EvidenceRequirement,
    StrictModel,
    Verdict,
)
from paper_research.agents.research_agent.pair_id_verifier import (
    MAX_PAIRS,
    Confidence,
    build_pair_contract_verifier,
    strict_pair_json,
)
from paper_research.agents.research_agent.scoped_verifier import (
    ScopedPairContractAdapter,
    partition_scope,
    validate_partition,
)
from paper_research.evaluation.phase2b import digest, sha
from paper_research.evaluation.phase2c_r5 import SPAN_VERSION, span_map
from paper_research.providers.llm import LLMProviderError

PROTOCOL = "v1_3_grounded_positive_support"
GROUNDING_INSTRUCTION = (
    " For SUPPORTS, PARTIAL, or CONTRADICTS, return one or more span IDs from the "
    "supplied evidence that ground the judgment. For IRRELEVANT, return an empty span list."
)


def evidence_span_maps(payload):
    """Use the exact R5 mapper on all raw passage text; no semantic selection."""
    maps = {}
    for e in payload["evidence"]:
        eid = e["evidence_id"]
        if eid in maps:
            raise ValueError("DUPLICATE_EVIDENCE_ID")
        maps[eid] = span_map(e["text"])
    return maps


def guard_candidate_input(payload, mapping):
    """The candidate boundary permits only the existing task and complete public span maps."""
    if set(payload) != {"question", "requirements", "evidence", "instruction", "pair_identity_map"}:
        raise ValueError("CANDIDATE_INPUT_LEAKAGE_OR_UNKNOWN_FIELD")
    if payload["pair_identity_map"] != mapping.as_input():
        raise ValueError("INEXACT_PAIR_BINDINGS")
    allowed_req = set(EvidenceRequirement.model_fields)
    for r in payload["requirements"]:
        if set(r) != allowed_req:
            raise ValueError("CANDIDATE_REQUIREMENT_LEAKAGE_OR_UNKNOWN_FIELD")
    for e in payload["evidence"]:
        if set(e) != {"evidence_id", "paper_id", "page", "block_ids", "chunk_id", "text", "spans"}:
            raise ValueError("CANDIDATE_EVIDENCE_LEAKAGE_OR_UNKNOWN_FIELD")
        if e["spans"] != span_map(e["text"]):
            raise ValueError("SELECTED_OR_ALTERED_SPAN_MAP")
    rids = {r["requirement_id"] for r in payload["requirements"]}
    eids = {e["evidence_id"] for e in payload["evidence"]}
    if any(p.requirement_id not in rids or p.evidence_id not in eids for p in mapping.entries):
        raise ValueError("UNBOUND_CANDIDATE_PAIR")


class GroundedAssessmentBatch(StrictModel):
    a: list[tuple[StrictStr, Verdict, Confidence, list[StrictStr]]] = Field(
        min_length=1,
        max_length=MAX_PAIRS,
    )
    bindings: ClassVar[tuple] = ()

    @model_validator(mode="after")
    def validate_grounding(self):
        expected = {pid: set(ids) for pid, _eid, ids in self.bindings}
        ids = [r[0] for r in self.a]
        if len(ids) != len(set(ids)) or set(ids) != set(expected):
            raise ValueError("INEXACT_EXPECTED_PAIR_SET")
        for pid, verdict, _confidence, spans in self.a:
            if len(spans) != len(set(spans)) or set(spans) - expected[pid]:
                raise ValueError("UNKNOWN_WRONG_EVIDENCE_OR_DUPLICATE_SPAN")
            if verdict == "IRRELEVANT" and spans:
                raise ValueError("IRRELEVANT_REQUIRES_EMPTY_SPANS")
            if verdict != "IRRELEVANT" and not spans:
                raise ValueError("POSITIVE_OR_CONTRADICTS_REQUIRES_SPAN")
        return self


@lru_cache(maxsize=128)
def _grounded_schema(bindings):
    ids = tuple(pid for pid, _eid, _spans in bindings)
    if not 1 <= len(ids) <= MAX_PAIRS:
        raise ValueError("INVALID_PAIR_COUNT")
    row = tuple[Literal[ids], Verdict, Confidence, list[StrictStr]]
    schema = create_model(
        f"GroundedAssessmentBatch{len(ids)}",
        __base__=GroundedAssessmentBatch,
        a=(
            list[row],
            Field(
                min_length=len(ids),
                max_length=len(ids),
                description="Every supplied pair once: "
                "[pair_id, verdict, confidence, support_span_ids].",
            ),
        ),
    )
    schema.bindings = bindings
    return schema


def grounded_schema(mapping, maps):
    return _grounded_schema(
        tuple(
            (p.pair_id, p.evidence_id, tuple(s["span_id"] for s in maps[p.evidence_id]))
            for p in mapping.entries
        )
    )


@dataclass(frozen=True)
class GroundedDecode:
    canonical_payload: dict
    assessments: tuple
    grounding: tuple
    raw_schema_valid: bool
    singleton_canonicalized: bool


def decode_grounded(raw, mapping, maps):
    data = strict_pair_json(raw) if isinstance(raw, str) else copy.deepcopy(raw)
    schema = grounded_schema(mapping, maps)
    canonicalized = False
    try:
        parsed = schema.model_validate(data)
        raw_valid = True
    except ValueError:
        if (
            not isinstance(data, dict)
            or set(data) != {"a"}
            or len(mapping.entries) != 1
            or not isinstance(data["a"], list)
            or len(data["a"]) != 4
            or data["a"][0] != mapping.expected_ids[0]
        ):
            raise
        canonical = {"a": [copy.deepcopy(data["a"])]}
        parsed = schema.model_validate(canonical)
        assert canonical["a"][0] == data["a"]
        data, raw_valid, canonicalized = canonical, False, True
    pairs = {p.pair_id: p for p in mapping.entries}
    assessments, grounding = [], []
    for pid, v, c, ids in parsed.a:
        p = pairs[pid]
        assessments.append(
            EvidenceAssessment(
                requirement_id=p.requirement_id,
                evidence_id=p.evidence_id,
                verdict=v,
                confidence=c,
                rationale=RATIONALE_NOT_REQUESTED,
            )
        )
        lookup = {s["span_id"]: s for s in maps[p.evidence_id]}
        grounding.append(
            {
                "pair_id": pid,
                "requirement_id": p.requirement_id,
                "evidence_id": p.evidence_id,
                "verdict": v,
                "confidence": c,
                "support_span_ids": list(ids),
                "cited_raw_spans": [copy.deepcopy(lookup[s]) for s in ids],
            }
        )
    return GroundedDecode(data, tuple(assessments), tuple(grounding), raw_valid, canonicalized)


class GroundedRawBoundary:
    """One provider dispatch per inherited attempt. No transport retry or semantic repair."""

    def __init__(self, provider, mapping, maps, payload, emit):
        self.provider, self.mapping, self.maps, self.payload = provider, mapping, maps, payload
        self.emit = emit

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def generate_structured_json(self, **kwargs):
        guard_candidate_input(self.payload, self.mapping)
        sent = json.loads(kwargs["user_prompt"])
        if sent["input"] != self.payload:
            raise ValueError("CANDIDATE_INPUT_DRIFT")
        guard_candidate_input(sent["input"], self.mapping)
        kwargs = {**kwargs, "system_prompt": kwargs["system_prompt"] + GROUNDING_INSTRUCTION}
        context = kwargs["request_context"]
        diag = {
            "protocol": PROTOCOL,
            "attempt": context["attempt_number"],
            "raw_schema_valid": False,
            "accepted_result": False,
            "singleton_canonicalized": False,
            "grounding_integrity": False,
            "span_map_hash": digest(self.maps),
            "span_mapper_version": SPAN_VERSION,
        }
        try:
            result = self.provider.generate_structured_json(**kwargs)
        except Exception:
            self.emit(diag)
            raise
        try:
            if len(result.usage_records) != 1:
                raise ValueError("EXACT_RAW_RESPONSE_REQUIRED")
            record = result.usage_records[0]
            diag.update(
                finish_reason=record.finish_reason, raw_response_path=record.raw_response_path
            )
            if record.finish_reason == "length" or not record.raw_response_path:
                raise ValueError("TRUNCATED_OR_MISSING_RAW_RESPONSE")
            diag["raw_response_sha256"] = sha(Path(record.raw_response_path))
            saved = json.loads(Path(record.raw_response_path).read_text(encoding="utf-8"))
            raw = strict_pair_json(saved["raw_content"])
            if raw != result.payload:
                raise ValueError("RAW_PARSED_RESPONSE_MISMATCH")
            decoded = decode_grounded(raw, self.mapping, self.maps)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            diag["validation_error"] = str(exc)[:400]
            self.emit(diag)
            raise LLMProviderError(
                str(exc),
                error_code="STRUCTURED_JSON_RESPONSE_ERROR",
                api_request_count=result.request_attempt_count,
                usage_records=result.usage_records,
            ) from exc
        diag.update(
            raw_schema_valid=decoded.raw_schema_valid,
            accepted_result=True,
            singleton_canonicalized=decoded.singleton_canonicalized,
            grounding_integrity=True,
            grounding=list(decoded.grounding),
            raw_semantic_records=copy.deepcopy(raw["a"]),
            canonical_semantic_records=copy.deepcopy(decoded.canonical_payload["a"]),
        )
        self.emit(diag)
        return result.model_copy(update={"payload": decoded.canonical_payload})


class GroundedPositiveSupportAdapter(ScopedPairContractAdapter):
    def __init__(self, provider, authority):
        StructuredContractAdapter.__init__(self, provider)
        self.authority = authority
        self.last_diagnostics = []

    def assess(self, state, requirement_ids, evidence_ids, runner):
        part = partition_scope(state, requirement_ids, evidence_ids, self.authority)
        trace = part.as_trace()
        state.tool_history.append({"scope_eligibility": trace})
        runner.trace.append(state, phase="SCOPE_ELIGIBILITY", extra=trace)
        self.last_diagnostics = []
        if not part.eligible:
            return []
        payload = assessment_input(state, requirement_ids, evidence_ids)
        active = {eid for _, eid in part.eligible}
        payload["evidence"] = [e for e in payload["evidence"] if e["evidence_id"] in active]
        payload["instruction"] = "Return exactly one assessment per listed eligible pair."
        mapping = part.mapping()
        payload["pair_identity_map"] = mapping.as_input()
        maps = evidence_span_maps(payload)
        for e in payload["evidence"]:
            e["spans"] = copy.deepcopy(maps[e["evidence_id"]])
        guard_candidate_input(payload, mapping)

        def emit(diag):
            self.last_diagnostics.append(copy.deepcopy(diag))
            state.tool_history.append({"grounded_verifier": copy.deepcopy(diag)})
            runner.trace.append(state, phase="GROUNDING_VALIDATION", extra=diag)

        provider = self.provider
        self.provider = GroundedRawBoundary(provider, mapping, maps, payload, emit)
        try:
            batch = self._request(
                state, runner, grounded_schema(mapping, maps), payload, "VERIFIER_FAILURE"
            )
        finally:
            self.provider = provider
        decoded = decode_grounded(batch.model_dump(mode="json"), mapping, maps)
        validate_partition(decoded.assessments, part)
        # Keep the existing scoped controller's final validation handle at its original boundary.
        state.tool_history.append({"scope_eligibility": trace})
        return list(decoded.assessments)


def build_grounded_contract_verifier(provider, *, verifier_protocol="v0", authority=None):
    """Separate explicit factory: no legacy factory, controller, registry or default changes."""
    if verifier_protocol in (PROTOCOL, "v1_2_compact_pair_id_scoped"):
        if authority is None:
            raise ValueError("EXPLICIT_AUTHORITATIVE_CORPUS_REQUIRED")
        cls = (
            GroundedPositiveSupportAdapter
            if verifier_protocol == PROTOCOL
            else ScopedPairContractAdapter
        )
        return cls(provider, authority)
    return build_pair_contract_verifier(provider, verifier_protocol=verifier_protocol)
