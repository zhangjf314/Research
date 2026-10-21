"""Opt-in V1_2: authoritative source eligibility before the frozen V1_1 codec."""

from dataclasses import dataclass
from enum import StrEnum
from types import FunctionType

from paper_research.agents.research_agent.agentic_retrieval import (
    EvidenceDrivenRetrievalController,
)
from paper_research.agents.research_agent.compact_verifier import assessment_input
from paper_research.agents.research_agent.contract_verifier import AgenticFailure
from paper_research.agents.research_agent.pair_id_verifier import (
    PairIdentity,
    PairIdentityContractAdapter,
    PairIdentityMap,
    decode_pairs,
    pair_schema,
)


class ScopeEligibility(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE_SOURCE_SCOPE = "INELIGIBLE_SOURCE_SCOPE"


def unresolved(detail):
    raise AgenticFailure("VERIFIER_FAILURE", "SOURCE_SCOPE_IDENTITY_UNRESOLVED: " + detail)


@dataclass(frozen=True)
class ScopePartition:
    eligible: tuple[tuple[str, str], ...]
    excluded: tuple[dict, ...]
    raw_count: int

    def mapping(self):
        if not self.eligible:
            return None
        return PairIdentityMap(
            tuple(PairIdentity(f"p{i:02d}", r, e) for i, (r, e) in enumerate(self.eligible))
        )

    def as_trace(self):
        return {
            "protocol": "V1_2_COMPACT_PAIR_ID_SCOPED",
            "raw_pair_count": self.raw_count,
            "eligible_pair_count": len(self.eligible),
            "filtered_pair_count": len(self.excluded),
            "excluded_pairs": list(self.excluded),
            "outcome": "ELIGIBLE_PAIRS_PENDING_SEMANTICS"
            if self.eligible
            else "DETERMINISTIC_NO_ELIGIBLE_PAIRS",
            "semantic_success": False,
        }


def partition_scope(state, requirement_ids, evidence_ids, authority):
    """Resolve every identity before evaluating scope, including excluded evidence."""
    if state.evidence_contract is None:
        unresolved("missing contract")
    reqs = [r for r in state.evidence_contract.requirements if r.requirement_id in requirement_ids]
    if (
        len(set(requirement_ids)) != len(requirement_ids)
        or {r.requirement_id for r in reqs} != set(requirement_ids)
        or len(set(evidence_ids)) != len(evidence_ids)
    ):
        unresolved("unknown or duplicate requested identity")
    papers = {c["paper_id"] for c in authority.values()}
    if any(set(r.source_scope) - papers for r in reqs):
        unresolved("unknown source_scope paper")
    by_id = {e.evidence_id: e for e in state.evidence_state.items.values()}
    identities = {}
    for eid in evidence_ids:
        e = by_id.get(eid)
        c = authority.get(e.chunk_id) if e is not None else None
        if c is None or (e.paper_id, e.page, e.block_ids, e.text_or_reference) != (
            c["paper_id"],
            c["page_start"],
            c["block_ids"],
            c["chunk_text"],
        ):
            # Fixtures use the same canonical prefix as the actual verifier input.
            if c is None or (e.paper_id, e.page, e.block_ids, e.text_or_reference) != (
                c["paper_id"],
                c["page_start"],
                c["block_ids"],
                c["chunk_text"][: state.agentic_config.evidence_chars],
            ):
                unresolved("unbound or mismatched evidence " + eid)
        p = state.evidence_provenance.get(eid)
        expected = {
            "paper_id": c["paper_id"],
            "chunk_id": c["chunk_id"],
            "page_start": c["page_start"],
            "block_ids": c["block_ids"],
        }
        if p is None or any(p.get(k) != v for k, v in expected.items()):
            unresolved("missing or mismatched provenance " + eid)
        identities[eid] = expected
    eligible, excluded = [], []
    for r in reqs:
        for eid in evidence_ids:
            if not r.source_scope or identities[eid]["paper_id"] in r.source_scope:
                eligible.append((r.requirement_id, eid))
            else:
                excluded.append(
                    {
                        "requirement_id": r.requirement_id,
                        "evidence_id": eid,
                        "eligibility": ScopeEligibility.INELIGIBLE_SOURCE_SCOPE.value,
                        "source_scope": list(r.source_scope),
                        "source_identity": identities[eid],
                        "reason": "authoritative paper_id outside nonempty source_scope",
                    }
                )
    return ScopePartition(tuple(eligible), tuple(excluded), len(reqs) * len(evidence_ids))


def validate_partition(assessments, partition):
    keys = [(a.requirement_id, a.evidence_id) for a in assessments]
    if len(keys) != len(set(keys)) or set(keys) != set(partition.eligible):
        raise AgenticFailure("VERIFIER_FAILURE", "eligible assessment pair set mismatch")


class ScopedPairContractAdapter(PairIdentityContractAdapter):
    def __init__(self, provider, authority):
        super().__init__(provider)
        self.authority = authority

    def assess(self, state, requirement_ids, evidence_ids, runner):
        partition = partition_scope(state, requirement_ids, evidence_ids, self.authority)
        trace = partition.as_trace()
        # Existing serialized state field; no new scientific state or verdicts.
        state.tool_history.append({"scope_eligibility": trace})
        runner.trace.append(state, phase="SCOPE_ELIGIBILITY", extra=trace)
        if not partition.eligible:
            return []
        payload = assessment_input(state, requirement_ids, evidence_ids)
        active_evidence = {eid for _, eid in partition.eligible}
        payload["evidence"] = [
            e for e in payload["evidence"] if e["evidence_id"] in active_evidence
        ]
        payload["instruction"] = "Return exactly one assessment per listed eligible pair."
        mapping = partition.mapping()
        payload["pair_identity_map"] = mapping.as_input()
        batch = self._request(
            state, runner, pair_schema(len(mapping.entries)), payload, "VERIFIER_FAILURE"
        )
        assessments = decode_pairs(batch, mapping)
        validate_partition(assessments, partition)
        return assessments


def _eligible_controller_validation(state, assessments, requirement_ids, evidence_ids):
    # The adapter resolved immutable authority and serialized the complete exclusion boundary.
    trace = state.tool_history[-1].get("scope_eligibility", {})
    excluded = {(p["requirement_id"], p["evidence_id"]) for p in trace.get("excluded_pairs", [])}
    expected = {(r, e) for r in requirement_ids for e in evidence_ids} - excluded
    if trace.get("raw_pair_count") != len(requirement_ids) * len(evidence_ids):
        unresolved("missing controller eligibility trace")
    validate_partition(assessments, ScopePartition(tuple(expected), (), len(expected)))


# Bind only the validation boundary in a private globals dictionary. All coverage, gap,
# replan, information-gain, budget, stop and reporting bytecode remains the frozen code.
_original_assess = EvidenceDrivenRetrievalController._assess
_scoped_assess = FunctionType(
    _original_assess.__code__,
    {**_original_assess.__globals__, "validate_assessments": _eligible_controller_validation},
    _original_assess.__name__,
    _original_assess.__defaults__,
    _original_assess.__closure__,
)


class ScopedEvidenceDrivenRetrievalController(EvidenceDrivenRetrievalController):
    _assess = _scoped_assess

    def __init__(self, runner, planner, verifier):
        if not isinstance(verifier, ScopedPairContractAdapter):
            raise ValueError("V1_2 controller requires authoritative scoped adapter")
        super().__init__(runner, planner, verifier)
