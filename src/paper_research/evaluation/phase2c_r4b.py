"""Truth-only human adjudication and outcome-independent requirement closure.

All protocol comparisons use frozen historical labels. This module does not
request models or modify verifier candidates. Human labels are never synthesized.
"""

import csv
import json
import random
from collections import Counter
from pathlib import Path

from pydantic import StrictStr

from paper_research.agents.research_agent.evidence_contract import StrictModel
from paper_research.evaluation.phase2c_r4 import Truth
from paper_research.evaluation.phase2c_r4a import MATERIAL_MARGIN, decision_case

BLIND_SEED = 20261006
LABEL_FIELDS = ("case_id", "truth", "quote")
POSITIVE = {"SUPPORTS", "PARTIAL", "CONTRADICTS"}


class TruthLabel(StrictModel):
    case_id: StrictStr
    truth: Truth
    quote: StrictStr


def configured_candidates(inventory):
    """A provider catalog entry is not an already configured generation model."""
    candidates = []
    for entry in inventory:
        model = entry.get("model") or ""
        family = model.casefold()
        if (
            entry.get("credential_present")
            and entry.get("provider") in ("siliconflow", "openai_compatible")
            and entry.get("base_url")
            and model
            and not any(x in family for x in ("deepseek", "qwen", "template"))
        ):
            candidates.append(entry)
    return sorted(candidates, key=lambda x: (x["config_path"], x["model"]))


def select_closure(subset, pairs, requirements, *, use_r4a_truth=False):
    """Minimal fixed closure needed to resolve each differing-status requirement.

    The historical 33-pair subset is mandatory. If its previously resolved truth is
    excluded by preregistration, reopen its requirement too, without choosing labels
    based on their values. Equal-status requirements cannot affect the paired primary
    delta and are excluded. Equal-verdict pairs INSIDE a differing-status requirement
    remain necessary for aggregation, despite their zero direct pair-delta effect.
    """
    pair_map = {r["case_id"]: r for r in pairs}
    if len(pair_map) != len(pairs):
        raise ValueError("DUPLICATE_HISTORICAL_PAIR")
    original_keys = {tuple(k) for k in subset["original_status_disagreement_requirements"]}
    source_ids = {r["case_id"] for r in subset["unknown_pairs"]}
    if len(source_ids) != subset["count"] or not source_ids <= pair_map.keys():
        raise ValueError("INEXACT_SELECTION_SOURCE")
    for row in subset["unknown_pairs"]:
        if row != pair_map[row["case_id"]]:
            raise ValueError("SELECTION_SOURCE_MUTATION")
    canonical_known = {r["case_id"] for r in pairs if use_r4a_truth and r["truth"] != "UNRESOLVED"}
    keys = {
        (r["item_id"], r["requirement_id"])
        for r in requirements
        if r["original_status_disagreement"] and r["truth"] == "UNRESOLVED"
    }
    if keys != original_keys:
        raise ValueError("INEXACT_SELECTION_REQUIREMENTS")
    if not use_r4a_truth:
        keys |= {
            (r["item_id"], r["requirement_id"])
            for r in requirements
            if r["original_status_disagreement"]
        }
    required = {r["case_id"] for r in pairs if (r["item_id"], r["requirement_id"]) in keys}
    selected = required - canonical_known
    if not source_ids <= selected:
        raise ValueError("SELECTION_SOURCE_NOT_COVERED")
    req_map = {(r["item_id"], r["requirement_id"]): r for r in requirements}
    records = []
    for pid in sorted(selected):
        pair = pair_map[pid]
        key = pair["item_id"], pair["requirement_id"]
        req = req_map[key]
        if req["v1_1"] == req["v1_2"]:
            raise ValueError("DIRECTION_IRRELEVANT_REQUIREMENT")
        records.append(
            {
                "canonical_input_id": key[0],
                "requirement_id": key[1],
                "pair_id": pid,
                "V1_1_verdict": pair["v1_1"],
                "V1_2_verdict": pair["v1_2"],
                "currently_unresolved": True,
                "historically_R4A_unresolved": pair["truth"] == "UNRESOLVED",
                "selection_origin": "FROZEN_UNRESOLVED_SUBSET"
                if pid in source_ids
                else "REOPEN_PRIOR_INDEPENDENT_REQUIREMENT_BY_TRUTH_SOURCE_POLICY",
                "direct_pair_delta_relevant": pair["v1_1"] != pair["v1_2"],
                "why_direction_relevant": (
                    "Eligible pair can change a differing-status requirement's truth; "
                    "include full unresolved aggregation closure, irrespective of pair labels."
                ),
                "which_decision_bound_it_can_change": [
                    "original_status_disagreements.full_subset_delta_bounds",
                    "all_target_requirements.full_subset_delta_bounds",
                ],
                "requirement_protocol_statuses": [req["v1_1"], req["v1_2"]],
                "full_denominator_bound_step": {
                    "original_status_disagreements": 1 / 17,
                    "all_target_requirements": 1 / 37,
                },
            }
        )
    closure = [
        {
            "canonical_input_id": sid,
            "requirement_id": rid,
            "pair_ids": sorted(
                r["case_id"] for r in pairs if (r["item_id"], r["requirement_id"]) == (sid, rid)
            ),
            "selected_pair_ids": sorted(
                r["case_id"]
                for r in pairs
                if (r["item_id"], r["requirement_id"]) == (sid, rid) and r["case_id"] in selected
            ),
        }
        for sid, rid in sorted(keys)
    ]
    return records, closure


def verify_closure(selection, closure, pairs, required_eligible):
    ids = {r["pair_id"] for r in selection}
    if len(ids) != len(selection):
        raise ValueError("DUPLICATE_SELECTED_PAIR")
    mapping = {r["case_id"]: r for r in pairs}
    for requirement in closure:
        key = requirement["canonical_input_id"], requirement["requirement_id"]
        included = {
            (mapping[p]["requirement_id"], mapping[p]["evidence_id"])
            for p in requirement["pair_ids"]
        }
        if included != set(required_eligible[key]):
            raise ValueError("INCOMPLETE_ELIGIBLE_REQUIREMENT_CLOSURE")
        if not set(requirement["selected_pair_ids"]) <= ids:
            raise ValueError("CLOSURE_PAIR_NOT_SELECTED")


def blind_cases(selection, source_pairs, seed=BLIND_SEED):
    order = sorted(r["pair_id"] for r in selection)
    rng = random.Random(seed)
    rng.shuffle(order)
    assignments = order[:]
    rng.shuffle(assignments)
    flip = {pid: bool(i % 2) for i, pid in enumerate(assignments)}
    public, private = [], {}
    for ordinal, pid in enumerate(order, 1):
        case_id = f"H{ordinal:03}"
        pair = source_pairs[pid]
        a, b = ("v1_2", "v1_1") if flip[pid] else ("v1_1", "v1_2")
        public.append(
            {
                "case_id": case_id,
                "scientific_requirement": pair["requirement"],
                "evidence_passage": pair["passage"],
                "authoritative_source_metadata": pair["metadata"],
                "judgment_A": pair[a],
                "judgment_B": pair[b],
            }
        )
        private[case_id] = {
            "pair_id": pid,
            "canonical_input_id": pair["item_id"],
            "requirement_id": pair["requirement_id"],
            "A_protocol": a,
            "B_protocol": b,
        }
    return public, private


def preference(truth, a, b):
    if truth == "UNRESOLVED":
        return "UNRESOLVED"
    if truth == a == b:
        return "BOTH"
    if truth == a:
        return "A"
    if truth == b:
        return "B"
    return "NEITHER"


def validate_labels(data, cases):
    expected = {c["case_id"]: c for c in cases}
    found = {}
    for value in data:
        label = TruthLabel.model_validate(value)
        if label.case_id not in expected:
            raise ValueError("UNKNOWN_CASE_ID")
        if label.case_id in found:
            raise ValueError("DUPLICATE_CASE_LABEL")
        if label.truth in POSITIVE and not label.quote:
            raise ValueError("POSITIVE_TRUTH_REQUIRES_QUOTE")
        if label.quote and label.quote not in expected[label.case_id]["evidence_passage"]:
            raise ValueError("QUOTE_NOT_EXACT_SUBSTRING")
        found[label.case_id] = label.model_dump()
    if found.keys() != expected.keys():
        raise ValueError("MISSING_REQUIRED_CASE_LABEL")
    return [found[c["case_id"]] for c in cases]


def load_submission(path):
    path = Path(path)
    if path.suffix.casefold() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(LABEL_FIELDS):
                raise ValueError("INEXACT_CSV_FIELDS")
            values = list(reader)
    elif path.suffix.casefold() == ".json":
        values = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(values, list):
            raise ValueError("JSON_LABELS_MUST_BE_ARRAY")
    elif path.suffix.casefold() == ".jsonl":
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    else:
        raise ValueError("UNSUPPORTED_SUBMISSION_FORMAT")
    return values


def combine_reviewers(first, second=None):
    if second is None:
        return [dict(r, review_status="SINGLE_HUMAN_REVIEW") for r in first]
    other = {r["case_id"]: r for r in second}
    if len(other) != len(second) or set(other) != {r["case_id"] for r in first}:
        raise ValueError("INEXACT_DOUBLE_REVIEW_CASE_SET")
    combined = []
    for a in first:
        b = other[a["case_id"]]
        if a["truth"] == b["truth"]:
            combined.append(
                dict(a, review_status="DOUBLE_HUMAN_AGREEMENT", second_quote=b["quote"])
            )
        else:
            combined.append(
                {
                    "case_id": a["case_id"],
                    "truth": "UNRESOLVED",
                    "quote": "",
                    "review_status": "HUMAN_DISAGREEMENT",
                    "reviewer_labels": [a, b],
                }
            )
    return combined


def require_complete_review(labels, cases):
    if {r["case_id"] for r in labels} != {r["case_id"] for r in cases}:
        raise ValueError("OPTIONAL_STOPPING_OR_PARTIAL_IMPORT_FORBIDDEN")


def closure_decision(primary, *, full_review_submitted, margin=MATERIAL_MARGIN):
    """Preserve R4A statistical case rules; R4B readiness is whole frozen closure.

    R4A's 39-of-41 *package output* floor is historical and stays failed. R4B's
    explicit pair-wise/human task supersedes package dispatch for this new panel;
    only complete frozen submissions reach the unchanged co-primary A/B/C math.
    Neither UNRESOLVED nor reviewer disagreement is repaired for readiness.
    """
    if not full_review_submitted:
        return "C", "AWAITING_HUMAN_ADJUDICATION"
    return decision_case(primary, availability=39, margin=margin)


def blind_balance(private):
    return dict(Counter(v["A_protocol"] for v in private.values()))
