"""R2 deterministic contract controls; no synthetic model improvement claims."""

import hashlib
from copy import deepcopy
from types import SimpleNamespace

import pytest

from paper_research.agents.research_agent.semantic_review import (
    StructuredSemanticVerifier,
    normalize_propositional_judgments,
    semantic_verification_input,
)
from paper_research.providers.llm import StructuredJSONResult, _deepseek_structured_tool
from tests.test_semantic_propositions import response, sample


def controls(claim):
    return [
        {
            "contract": "claim-continuation-prefix-v1",
            "start": 0,
            "end": 10,
            "text": "[CONTINUE]",
            "claim_sha256": hashlib.sha256(claim.encode()).hexdigest(),
        }
    ]


def r2_response(p, first="SUPPORTED", second="INSUFFICIENT", declared="PARTIAL"):
    r = response(p, first, second, declared)
    for i, part in enumerate(r["judgments"][0]["propositions"]):
        part.update(
            proposition_id=f"p{i}", depends_on=[], prerequisites_complete=True, atomicity="ATOMIC"
        )
    return r


def normalized(p, r):
    output, audit = normalize_propositional_judgments(p, r, require_dependencies=True)
    return output["judgments"][0], audit[0]


def test_only_explicit_original_text_bound_control_is_exempted():
    p = sample()
    p["claims"][0]["text"] = "[CONTINUE] " + p["claims"][0]["text"]
    r = r2_response(p)
    assert normalized(p, r)[0]["unable_reason"] == "PROPOSITION_CLAIM_COVERAGE_INCOMPLETE"
    p["claims"][0]["format_controls"] = controls(p["claims"][0]["text"])
    before = deepcopy(p)
    row, audit = normalized(p, r)
    assert row["verdict"] == "PARTIAL" and audit["format_controls"][0]["start"] == 0
    assert p == before  # Full original text and character positions are preserved.
    projected = semantic_verification_input(p, include_format_controls=True)
    claim = projected["evidence_scopes"][0]["claims"][0]
    assert claim["text"].startswith("[CONTINUE]") and claim["format_controls"]
    assert audit["propositions"][0]["start"] == 11


@pytest.mark.parametrize(
    "field,value",
    [
        ("start", 1),
        ("end", 30),
        ("text", "In general"),
        ("claim_sha256", "0" * 64),
        ("contract", "untrusted"),
    ],
)
def test_format_declaration_cannot_skip_semantic_text(field, value):
    p = sample()
    p["claims"][0]["text"] = "[CONTINUE] " + p["claims"][0]["text"]
    p["claims"][0]["format_controls"] = controls(p["claims"][0]["text"])
    p["claims"][0]["format_controls"][0][field] = value
    assert normalized(p, r2_response(p))[0]["unable_reason"] == "INVALID_CLAIM_FORMAT_CONTROLS"
    with pytest.raises(ValueError):
        semantic_verification_input(p, include_format_controls=True)


@pytest.mark.parametrize("qualifier", ["In general, ", "only ", "always ", "not always ", "if "])
def test_scope_quantifier_negation_and_condition_cannot_be_omitted(qualifier):
    p = sample()
    p["claims"][0]["text"] = qualifier + p["claims"][0]["text"]
    assert normalized(p, r2_response(p))[0]["unable_reason"] == (
        "PROPOSITION_CLAIM_COVERAGE_INCOMPLETE"
    )


def test_model_cannot_declare_its_own_coverage_exemption():
    p = sample()
    p["claims"][0]["text"] = "[CONTINUE] " + p["claims"][0]["text"]
    r = r2_response(p)
    r["judgments"][0]["format_controls"] = controls(p["claims"][0]["text"])
    assert normalized(p, r)[0]["verdict"] == "INSUFFICIENT"


@pytest.mark.parametrize("dependent_verdict", ["SUPPORTED", "CONTRADICTED"])
def test_missing_identity_blocks_both_dependent_support_and_conflict(dependent_verdict):
    p = sample()
    r = r2_response(p, "INSUFFICIENT", dependent_verdict, "CONTRADICTED")
    r["judgments"][0]["propositions"][1]["depends_on"] = ["p0"]
    row, audit = normalized(p, r)
    assert row["verdict"] == "INSUFFICIENT"
    assert audit["propositions"][1]["dependency_blocked_by"] == ["p0"]
    assert "blocked_by=['p0']" in row["rationale"]


def test_transitive_dependency_chain_cannot_borrow_an_unproved_premise():
    p = sample()
    p["claims"][0]["text"] = "Identity. Scope. Comparison."
    r = r2_response(p, "INSUFFICIENT", "SUPPORTED", "CONTRADICTED")
    parts = r["judgments"][0]["propositions"]
    parts.append(deepcopy(parts[1]))
    for i, (q, quote) in enumerate(zip(parts, ["Identity.", "Scope.", "Comparison."], strict=True)):
        q.update(claim_quote=quote, proposition_id=f"p{i}", depends_on=[f"p{i - 1}"] if i else [])
    parts[2]["verdict"] = "CONTRADICTED"
    row, audit = normalized(p, r)
    assert row["verdict"] == "INSUFFICIENT"
    assert audit["propositions"][2]["dependency_blocked_by"] == ["p1"]


def test_supported_premise_allows_comparison_and_preserves_conflict_priority():
    p = sample()
    r = r2_response(p, "SUPPORTED", "CONTRADICTED", "SUPPORTED")
    r["judgments"][0]["propositions"][1]["depends_on"] = ["p0"]
    assert normalized(p, r)[0]["verdict"] == "CONTRADICTED"
    r = r2_response(p, "SUPPORTED", "SUPPORTED", "SUPPORTED")
    r["judgments"][0]["propositions"][1]["depends_on"] = ["p0"]
    assert normalized(p, r)[0]["verdict"] == "SUPPORTED"


def test_forward_reference_partial_and_independent_conflict_are_not_blanket_abstentions():
    p = sample()
    r = r2_response(p, "CONTRADICTED", "SUPPORTED", "PARTIAL")
    r["judgments"][0]["propositions"][0]["depends_on"] = ["p1"]
    assert normalized(p, r)[0]["verdict"] == "CONTRADICTED"
    r = r2_response(p)
    r["judgments"][0]["propositions"][1]["prerequisites_complete"] = False
    row, _ = normalized(p, r)
    assert row["verdict"] == "PARTIAL" and row["supported_parts"] and row["unsupported_parts"]
    r = r2_response(p, "CONTRADICTED", "SUPPORTED", "SUPPORTED")
    r["judgments"][0]["propositions"][1]["prerequisites_complete"] = False
    assert normalized(p, r)[0]["verdict"] == "CONTRADICTED"


@pytest.mark.parametrize(
    "deps", [(["unknown"], []), (["p0"], []), (["p1"], ["p0"]), (["p1", "p1"], [])]
)
def test_invalid_dependency_graph_cannot_license_support_or_conflict(deps):
    p = sample()
    r = r2_response(p)
    for q, values in zip(r["judgments"][0]["propositions"], deps, strict=True):
        q["depends_on"] = values
    assert normalized(p, r)[0]["unable_reason"] == "PROPOSITION_DEPENDENCIES_INVALID"


def test_duplicate_ids_and_legacy_missing_contract_fail_closed():
    p = sample()
    r = r2_response(p)
    r["judgments"][0]["propositions"][1]["proposition_id"] = "p0"
    assert normalized(p, r)[0]["unable_reason"] == "PROPOSITION_DEPENDENCIES_INVALID"
    assert normalized(p, response(p))[0]["unable_reason"] == "PROPOSITION_CONTRACT_INVALID"


@pytest.mark.parametrize("atomicity", ["COMPOUND", "UNRESOLVED"])
def test_non_atomic_or_unresolved_parts_do_not_license_support(atomicity):
    p = sample()
    r = r2_response(p, "SUPPORTED", "SUPPORTED", "SUPPORTED")
    for q in r["judgments"][0]["propositions"]:
        q["atomicity"] = atomicity
    assert normalized(p, r)[0]["verdict"] == "INSUFFICIENT"


def test_dependency_blocks_do_not_hide_invalid_source_quotes():
    p = sample()
    r = r2_response(p)
    q = r["judgments"][0]["propositions"][0]
    q["depends_on"] = ["p1"]
    q["spans"][0]["evidence_id"] = "X"
    assert normalized(p, r)[0]["unable_reason"] == "EVIDENCE_OUTSIDE_CLAIM_SCOPE"


def test_r2_default_adapter_retains_trace_and_excludes_old_answers():
    p = sample()
    p["claims"][0]["text"] = "[CONTINUE] " + p["claims"][0]["text"]
    p["claims"][0]["format_controls"] = controls(p["claims"][0]["text"])
    p["gold"] = "DO_NOT_SEND"
    p["claims"][0]["old_ai_verdict"] = "DO_NOT_SEND"
    calls = []
    provider = SimpleNamespace(
        max_retries=0,
        generate_structured_json=lambda **kw: (
            calls.append(kw)
            or StructuredJSONResult(provider="offline", model="fixture", payload=r2_response(p))
        ),
    )
    verifier = StructuredSemanticVerifier(provider)
    assert verifier(p)["judgments"][0]["verdict"] == "PARTIAL"
    assert calls[0]["schema_name"] == "research-semantic-review-v4-r2"
    assert "DO_NOT_SEND" not in calls[0]["user_prompt"] and len(calls) == 1
    assert "claim-continuation-prefix-v1" in calls[0]["user_prompt"]
    audit = verifier.context["proposition_audit"][0]
    assert audit["propositions"][0]["dependency_blocked_by"] == []
    _, schema = _deepseek_structured_tool(calls[0]["schema_name"], {})
    item = schema["properties"]["judgments"]["items"]["properties"]["propositions"]["items"]
    assert {"depends_on", "proposition_id", "prerequisites_complete", "atomicity"} <= set(
        item["required"]
    )
    assert item["additionalProperties"] is False
