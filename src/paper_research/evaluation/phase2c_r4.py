"""Read-only stability/adjudication analysis; never modifies verifier protocols."""

from collections import Counter
from itertools import combinations
from typing import Literal

import numpy as np
from pydantic import Field, StrictStr, model_validator

from paper_research.agents.research_agent.contract_verifier import update_coverage
from paper_research.agents.research_agent.evidence_contract import (
    EvidenceAssessment,
    EvidenceContract,
    StrictModel,
)
from paper_research.agents.research_agent.models import EvidenceItem
from paper_research.agents.research_agent.state import AgentState
from paper_research.evaluation.phase2b import digest

ARMS = ("s11", "s12")
PROTOCOLS = {"s11": "v1_1_compact_pair_id", "s12": "v1_2_compact_pair_id_scoped"}
SEED = 20261002
ORDER_SEED = 20261004
REPEAT_ID = "r4-repeat-1"


def repeat_identity(item, arm, protocol_sha):
    if arm not in ARMS:
        raise ValueError("unknown repeat arm")
    return digest(
        {
            "canonical_input_hash": item["canonical_input_hash"],
            "arm": arm,
            "repeat_id": REPEAT_ID,
            "protocol_sha256": protocol_sha,
        }
    )


def balanced_order(items):
    import random

    items = items[:]
    random.Random(ORDER_SEED).shuffle(items)
    return [
        {"item_id": x["item_id"], "arms": list(ARMS if i % 2 == 0 else ARMS[::-1])}
        for i, x in enumerate(items)
    ]


def eligible_pairs(item):
    return [
        (r["requirement_id"], e["evidence_id"])
        for r in item["input"]["requirements"]
        for e in item["input"]["evidence"]
        if not r["source_scope"] or e["paper_id"] in r["source_scope"]
    ]


def assessment_map(result):
    rows = result["assessments"]
    mapping = {(a["requirement_id"], a["evidence_id"]): a for a in rows}
    if len(mapping) != len(rows):
        raise ValueError("duplicate assessment identity")
    return mapping


def statuses(item, assessments):
    state = AgentState(
        task_id="offline-" + item["item_id"], research_question=item["input"]["question"]
    )
    state.evidence_contract = EvidenceContract(
        contract_id="offline",
        question=state.research_question,
        created_by="diagnostic",
        requirements=item["input"]["requirements"],
    )
    for e in item["input"]["evidence"]:
        state.evidence_state.add(
            EvidenceItem(
                **{k: v for k, v in e.items() if k != "text"},
                block_id=e["chunk_id"],
                text_or_reference=e["text"],
                discovered_by_tool="diagnostic",
                discovered_at_step=0,
            ),
            [],
        )
    state.evidence_assessments = [EvidenceAssessment.model_validate(a) for a in assessments]
    update_coverage(state)
    return {rid: c.status.value for rid, c in state.requirement_coverage.items()}


def compare(item, left, right):
    if not (left["semantic_result_available"] and right["semantic_result_available"]):
        return None
    maps = [assessment_map(r) for r in (left, right)]
    pairs = eligible_pairs(item)
    if not all(set(pairs) <= m.keys() for m in maps):
        raise ValueError("eligible pair missing from legal result")
    a, b = [statuses(item, r["assessments"]) for r in (left, right)]
    return {
        "item_id": item["item_id"],
        "scope_stratum": item["scope_stratum"],
        "pair_count": item["pair_count"],
        "pair_denominator": len(pairs),
        "binary_matches": sum(
            (maps[0][k]["verdict"] == "SUPPORTS") == (maps[1][k]["verdict"] == "SUPPORTS")
            for k in pairs
        ),
        "exact_matches": sum(maps[0][k]["verdict"] == maps[1][k]["verdict"] for k in pairs),
        "status_denominator": len(a),
        "status_matches": sum(a[k] == b[k] for k in a),
        "status_disagreements": [k for k in a if a[k] != b[k]],
    }


def pooled(rows, numerator, denominator):
    total = sum(r[denominator] for r in rows)
    return sum(r[numerator] for r in rows) / total if total else None


def clustered_interval(rows, numerator, denominator):
    if not rows:
        return {"point": None, "ci95": None, "canonical_inputs": 0}
    n = np.array([r[numerator] for r in rows], dtype=float)
    d = np.array([r[denominator] for r in rows], dtype=float)
    indices = np.random.default_rng(SEED).integers(0, len(rows), (10000, len(rows)))
    samples = n[indices].sum(axis=1) / d[indices].sum(axis=1)
    return {
        "point": float(n.sum() / d.sum()),
        "ci95": [float(x) for x in np.percentile(samples, [2.5, 97.5])],
        "canonical_inputs": len(rows),
        "denominator": int(d.sum()),
        "unit": "canonical_input",
        "replicates": 10000,
        "seed": SEED,
        "estimator": "pooled micro agreement with canonical-input cluster resampling",
    }


def agreement_summary(rows):
    return {
        name: clustered_interval(rows, numerator, denominator)
        for name, numerator, denominator in [
            ("binary", "binary_matches", "pair_denominator"),
            ("exact", "exact_matches", "pair_denominator"),
            ("status", "status_matches", "status_denominator"),
        ]
    }


def cross_vs_self(cross, self11, self12):
    def disagreement(rs):
        value = pooled(rs, "status_matches", "status_denominator")
        return 1 - value if value is not None else None

    values = [disagreement(rs) for rs in (cross, self11, self12)]
    descriptive = {
        "cross_disagreement": values[0],
        "self11_disagreement": values[1],
        "self12_disagreement": values[2],
        "vs_self11": values[0] - values[1] if all(v is not None for v in values[:2]) else None,
        "vs_self12": values[0] - values[2]
        if values[0] is not None and values[2] is not None
        else None,
        "vs_mean_self": values[0] - (values[1] + values[2]) / 2
        if all(v is not None for v in values)
        else None,
        "interpretation": "descriptive only; unbalanced available-result populations disclosed",
    }
    maps = [{r["item_id"]: r for r in rs} for rs in (cross, self11, self12)]
    common = sorted(set.intersection(*(set(m) for m in maps)))
    if not common:
        return {"descriptive": descriptive, "common_four_legal": {"n": 0}}
    denominators = np.array([maps[0][k]["status_denominator"] for k in common], dtype=float)
    failures = [
        np.array([m[k]["status_denominator"] - m[k]["status_matches"] for k in common], dtype=float)
        for m in maps
    ]
    indices = np.random.default_rng(SEED).integers(0, len(common), (10000, len(common)))
    total = denominators[indices].sum(axis=1)
    contrasts = [
        failures[0] - failures[1],
        failures[0] - failures[2],
        failures[0] - (failures[1] + failures[2]) / 2,
    ]
    intervals = {}
    for name, contrast in zip(("vs_self11", "vs_self12", "vs_mean_self"), contrasts, strict=True):
        samples = contrast[indices].sum(axis=1) / total
        intervals[name] = {
            "point": float(contrast.sum() / denominators.sum()),
            "ci95": [float(x) for x in np.percentile(samples, [2.5, 97.5])],
        }
    return {
        "descriptive": descriptive,
        "common_four_legal": {
            "n": len(common),
            "item_ids": common,
            "requirements": int(denominators.sum()),
            "excess": intervals,
            "unit": "canonical_input",
            "seed": SEED,
            "replicates": 10000,
            "qualification_score": False,
        },
    }


Truth = Literal["SUPPORTS", "PARTIAL", "IRRELEVANT", "CONTRADICTS", "UNRESOLVED"]
Preference = Literal["A", "B", "BOTH", "NEITHER", "UNRESOLVED"]


class AdjudicationRow(StrictModel):
    pair_id: StrictStr
    support_strength_truth: Truth
    preferred_judgment: Preference
    evidence_quote: str = Field(max_length=400)
    reason: str = Field(min_length=1, max_length=500)


class AdjudicationBatch(StrictModel):
    judgments: list[AdjudicationRow] = Field(min_length=1, max_length=40)

    @model_validator(mode="after")
    def unique_pairs(self):
        if len({r.pair_id for r in self.judgments}) != len(self.judgments):
            raise ValueError("duplicate adjudication pair")
        return self


def validate_adjudication(data, package):
    parsed = AdjudicationBatch.model_validate(data)
    pairs = {p["pair_id"]: p for p in package["pairs"]}
    if {r.pair_id for r in parsed.judgments} != pairs.keys():
        raise ValueError("inexact adjudication pair set")
    evidence = {e["evidence_id"]: e["text"] for e in package["evidence"]}
    for row in parsed.judgments:
        pair = pairs[row.pair_id]
        a, b = pair["judgment_A"], pair["judgment_B"]
        truth = row.support_strength_truth
        expected = (
            "UNRESOLVED"
            if truth == "UNRESOLVED"
            else (
                "BOTH"
                if a == b == truth
                else "A"
                if a == truth
                else "B"
                if b == truth
                else "NEITHER"
            )
        )
        if row.preferred_judgment != expected:
            raise ValueError("truth/preference inconsistency")
        quote = " ".join(row.evidence_quote.split())
        if truth not in ("IRRELEVANT", "UNRESOLVED") and len(quote) < 20:
            raise ValueError("resolved positive/contradictory truth requires source quote")
        if quote and quote not in " ".join(evidence[pair["evidence_id"]].split()):
            raise ValueError("quote not grounded in supplied passage")
    return parsed


def grounded_statuses(item, truth, base_result, required_ids):
    """Conservative unresolved aggregates; never use a protocol label as an imputed truth."""
    pairs = eligible_pairs(item)
    out = {}
    base = assessment_map(base_result)
    for rid in required_ids:
        keys = [k for k in pairs if k[0] == rid]
        if any(truth.get(k, "UNRESOLVED") == "UNRESOLVED" for k in keys):
            out[rid] = "UNRESOLVED"
            continue
        rows = [{**base[k], "verdict": truth[k]} for k in keys]
        out[rid] = statuses(item, rows)[rid]
    return out


def accuracy(rows):
    resolved = [r for r in rows if r["truth"] != "UNRESOLVED"]
    return {
        "total": len(rows),
        "resolved_denominator": len(resolved),
        "unresolved": len(rows) - len(resolved),
        "v1_1_correct": sum(r["v1_1"] == r["truth"] for r in resolved),
        "v1_2_correct": sum(r["v1_2"] == r["truth"] for r in resolved),
        "v1_1_accuracy": sum(r["v1_1"] == r["truth"] for r in resolved) / len(resolved)
        if resolved
        else None,
        "v1_2_accuracy": sum(r["v1_2"] == r["truth"] for r in resolved) / len(resolved)
        if resolved
        else None,
    }


def grouped_accuracy(rows, axis):
    return {
        str(k): accuracy([r for r in rows if r[axis] == k]) for k in sorted({r[axis] for r in rows})
    }


def transition_counts(rows):
    return dict(Counter(r["v1_1"] + " -> " + r["v1_2"] for r in rows))


def accuracy_delta_interval(rows):
    # Preserve all pairs from an input together; resolved-only accuracy is explicitly conditional.
    by_input = {}
    for row in rows:
        if row["truth"] == "UNRESOLVED":
            continue
        entry = by_input.setdefault(row["item_id"], {"difference": 0, "denominator": 0})
        entry["difference"] += int(row["v1_2"] == row["truth"]) - int(row["v1_1"] == row["truth"])
        entry["denominator"] += 1
    return clustered_interval(list(by_input.values()), "difference", "denominator")


def pairwise_comparisons(panel):
    return {f"{a}__{b}": (panel[a], panel[b]) for a, b in combinations(panel, 2)}
