"""Frozen GLM-only truth schema, transport guards, and bounded case semantics."""

import copy
import json
import random

from pydantic import Field, StrictStr

from paper_research.agents.research_agent.evidence_contract import StrictModel
from paper_research.evaluation.phase2c_r4 import Truth
from paper_research.evaluation.phase2c_r4b import closure_decision

MODEL = "zai-org/GLM-5.2"
ENDPOINT = "https://api.siliconflow.cn/v1/chat/completions"
R4B_MANIFEST_SHA = "5fa9a14ce564c1826dd6d06854946f028f7a4524f83bc369cff04a6fc4000062"
ORDER_SEED = 20261006
CASE_FIELDS = (
    "case_id",
    "scientific_requirement",
    "evidence_passage",
    "authoritative_source_metadata",
)


class GroundedTruth(StrictModel):
    case_id: StrictStr
    truth: Truth
    quote: StrictStr = Field(max_length=400)


def strict_json_output(content):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("DUPLICATE_JSON_FIELD")
            result[key] = value
        return result

    return json.loads(content, object_pairs_hook=unique_object)


def exact_identity(provider, configured, returned=None, endpoint=ENDPOINT):
    if provider != "siliconflow" or configured != MODEL or endpoint != ENDPOINT:
        raise ValueError("GLM_EXACT_IDENTITY_OR_NO_FALLBACK_GUARD")
    if returned is not None and returned != MODEL:
        raise ValueError("GLM_RETURNED_IDENTITY_MISMATCH")


def manifest_identity(observed):
    if observed != R4B_MANIFEST_SHA:
        raise ValueError("PHASE2C_R4C_INVALID_INPUT_IDENTITY")


def case_set_guard(cases):
    if len(cases) != 34 or {c["case_id"] for c in cases} != {f"H{i:03}" for i in range(1, 35)}:
        raise ValueError("INEXACT_34_CASE_SET")


def scientific_input(case):
    """Only four allowlisted fields: never anonymous or protocol verdicts."""
    if isinstance(case, list):
        raise ValueError("ONE_PAIR_PER_REQUEST_REQUIRED")
    return {key: copy.deepcopy(case[key]) for key in CASE_FIELDS}


def validate_truth(data, case):
    parsed = GroundedTruth.model_validate(data)
    if parsed.case_id != case["case_id"]:
        raise ValueError("INEXACT_CASE_ID")
    if parsed.truth in ("SUPPORTS", "PARTIAL", "CONTRADICTS") and len(parsed.quote) < 20:
        raise ValueError("POSITIVE_QUOTE_LENGTH_OUT_OF_RANGE")
    if parsed.quote and parsed.quote not in case["evidence_passage"]:
        raise ValueError("QUOTE_NOT_RAW_EXACT_SUBSTRING")
    return parsed.model_dump()


def execution_order(cases):
    case_set_guard(cases)
    ids = sorted(c["case_id"] for c in cases)
    random.Random(ORDER_SEED).shuffle(ids)
    return ids


def explicit_schema_unsupported(status, response):
    text = str(response).casefold()
    named_format = "json_schema" in text or "response_format" in text
    explicit = any(
        x in text
        for x in (
            "not supported",
            "unsupported",
            "does not support",
            "not support",
            "input should be 'text' or 'json_object'",
            'input should be "text" or "json_object"',
        )
    )
    return status in (400, 404, 422) and named_format and explicit


def next_semantic_attempt(attempts):
    """Saved invalid attempts count; a saved valid result terminates requests."""
    if any(r["valid"] for r in attempts) or len(attempts) >= 2:
        return None
    return len(attempts) + 1


def phase_cost_guard(estimated, *, case_started=False):
    if estimated > 10 and not case_started:
        return "HARD_STOP_BEFORE_NEXT_CASE"
    return "SOFT_WARNING_CONTINUE" if estimated > 5 else "NORMAL"


def reference_cost(inputs, outputs):
    if not isinstance(inputs, int) or not isinstance(outputs, int):
        return None
    return (inputs * 8 + outputs * 28) / 1_000_000


def correctness(truth, first, second):
    if truth == "UNRESOLVED":
        return "UNRESOLVED"
    if first == second == truth:
        return "BOTH_CORRECT"
    if first == truth:
        return "V1_1_CORRECT"
    if second == truth:
        return "V1_2_CORRECT"
    return "NEITHER_CORRECT"


def canonical_glm_records(historical, case_to_pair, glm_labels):
    """No historic truth is canonical, including when GLM is missing."""
    truth = {case_to_pair[r["case_id"]]: r for r in glm_labels}
    output = []
    for previous in historical:
        row = {
            k: copy.deepcopy(v)
            for k, v in previous.items()
            if k not in ("truth", "independent_judgment", "judgment_available")
        }
        label = truth.get(row["case_id"])
        row.update(
            truth=label["truth"] if label else "UNRESOLVED",
            judgment_available=label is not None,
            canonical_truth_source="R4C_GLM_ONLY" if label else None,
        )
        row["code_correctness"] = correctness(row["truth"], row["v1_1"], row["v1_2"])
        output.append(row)
    return output


def scientific_decision(primary, all34_processed):
    if not all34_processed:
        return "C", "INDEPENDENT_ADJUDICATION_INCOMPLETE"
    return closure_decision(primary, full_review_submitted=True)
