"""Fresh absolute accuracy evaluation. No verifier or gold label intervention."""

import copy
import re
from collections import Counter
from typing import Literal

import numpy as np
from pydantic import StrictStr

from paper_research.agents.research_agent.evidence_contract import StrictModel
from paper_research.evaluation.phase2b import digest, normalize
from paper_research.evaluation.phase2c_r4 import eligible_pairs, statuses
from paper_research.evaluation.phase2c_r4c import next_semantic_attempt

CLASSES = ("SUPPORTS", "PARTIAL", "IRRELEVANT", "CONTRADICTS")
POSITIVE = frozenset(("SUPPORTS", "PARTIAL"))
SPAN_VERSION = "raw-boundaries-v1: punctuation-whitespace-or-newline; max600-codepoints"
GATES = {
    "minimum_inputs": 60,
    "gold_availability": 0.90,
    "first_valid": 0.98,
    "bounded_availability": 0.98,
    "truncation_max": 0.02,
    "scope_rejection_max": 0,
    "coverage_accuracy": 0.95,
    "exact_accuracy": 0.90,
    "status_accuracy": 0.90,
    "coverage_ci_floor": 0.90,
    "exact_ci_floor": 0.80,
    "status_ci_floor": 0.80,
    "false_support_max": 0.05,
    "class_minimum_n": 20,
    "class_recall": 0.75,
}


class SpanTruth(StrictModel):
    case_id: StrictStr
    truth: Literal["SUPPORTS", "PARTIAL", "IRRELEVANT", "CONTRADICTS", "UNRESOLVED"]
    support_span_ids: list[StrictStr]


def span_map(text):
    """Partition raw Python Unicode character offsets without rewriting any character."""
    ends = [m.end() for m in re.finditer(r"(?<=[.!?])\s+|\n+", text)]
    ends.append(len(text))
    spans, start = [], 0
    for end in sorted(set(ends)):
        while start < end:
            stop = min(end, start + 600)
            spans.append(
                {
                    "span_id": f"s{len(spans):02d}",
                    "start": start,
                    "end": stop,
                    "raw_text": text[start:stop],
                }
            )
            start = stop
    assert "".join(s["raw_text"] for s in spans) == text
    return spans


def validate_gold(data, case):
    parsed = SpanTruth.model_validate(data)
    if parsed.case_id != case["case_id"]:
        raise ValueError("INEXACT_CASE_ID")
    ids = parsed.support_span_ids
    if len(ids) != len(set(ids)) or set(ids) - {s["span_id"] for s in case["spans"]}:
        raise ValueError("INVALID_OR_DUPLICATE_SPAN_ID")
    if parsed.truth in ("SUPPORTS", "PARTIAL", "CONTRADICTS") and not ids:
        raise ValueError("MISSING_GROUNDING_SPAN")
    if parsed.truth == "IRRELEVANT" and ids:
        raise ValueError("IRRELEVANT_MUST_HAVE_EMPTY_SPANS")
    return parsed.model_dump()


def gold_input(case):
    # Construction metadata and private pair bindings never cross this boundary.
    return copy.deepcopy(
        {
            "gold_case_id": case["case_id"],
            "requirement": case["requirement"],
            "candidate_evidence": {"raw_passage": case["passage"], "spans": case["spans"]},
            "authoritative_provenance": case["provenance"],
        }
    )


def identities(payload, anchors=()):
    reqs, evidence = payload["requirements"], payload["evidence"]
    return {
        "exact_question": payload["question"],
        "normalized_question": normalize(payload["question"]),
        "exact_requirements": [digest(r["description"]) for r in reqs],
        "normalized_requirements": [normalize(r["description"]) for r in reqs],
        "requirement_sources": [
            digest([normalize(r["description"]), sorted(r["source_scope"]), r["min_sources"]])
            for r in reqs
        ],
        "evidence_set": digest(sorted(e["chunk_id"] for e in evidence)),
        "canonical_input": digest(payload),
        "pair_signatures": [
            digest(
                [
                    normalize(r["description"]),
                    sorted(r["source_scope"]),
                    e["chunk_id"],
                    digest(e["text"]),
                ]
            )
            for r in reqs
            for e in evidence
        ],
        "source_anchor_combination": digest(sorted(a["chunk_id"] for a in anchors))
        if anchors
        else None,
    }


def overlap(payload, anchors, prior):
    sig = identities(payload, anchors)
    out = []
    for key in (
        "exact_question",
        "normalized_question",
        "evidence_set",
        "canonical_input",
        "source_anchor_combination",
    ):
        if sig[key] is not None and sig[key] in prior[key]:
            out.append(key)
    for key in (
        "exact_requirements",
        "normalized_requirements",
        "requirement_sources",
        "pair_signatures",
    ):
        if set(sig[key]) & prior[key]:
            out.append(key)
    return out


def index_identity(prior, sig):
    for key, value in sig.items():
        if value is None:
            continue
        prior.setdefault(key, set()).update(value if isinstance(value, list) else [value])


def empty_index():
    return {k: set() for k in identities({"question": "", "requirements": [], "evidence": []})}


def gold_before_verifier(freeze, available, total):
    if not freeze or not total or available / total < GATES["gold_availability"]:
        raise ValueError("GOLD_BEFORE_VERIFIER_GUARD")


def gold_status(item, truths):
    pairs = eligible_pairs(item)
    result, assessments = {}, []
    for rid, eid in pairs:
        truth = truths.get((rid, eid))
        if truth in CLASSES:
            assessments.append(
                {
                    "requirement_id": rid,
                    "evidence_id": eid,
                    "verdict": truth,
                    "confidence": 1,
                    "rationale": "independent model gold",
                }
            )
    computed = statuses(item, assessments)
    for r in item["input"]["requirements"]:
        rid = r["requirement_id"]
        result[rid] = (
            computed[rid]
            if all(truths.get((r0, e)) in CLASSES for r0, e in pairs if r0 == rid)
            else "UNRESOLVED"
        )
    return result


def clustered(records, numerator="correct", denominator="n", delta=False):
    """Each row is a whole canonical input, including zero-denominator rows."""
    if not records or sum(r[denominator] for r in records) == 0:
        return {
            "point": None,
            "ci95": None,
            "denominator": 0,
            "clusters": len(records),
            "replicates": 10000,
            "seed": 20261002,
            "unit": "canonical_input",
        }
    a = np.array([r[numerator] for r in records], float)
    d = np.array([r[denominator] for r in records], float)
    idx = np.random.default_rng(20261002).integers(0, len(records), (10000, len(records)))
    denominators = d[idx].sum(axis=1)
    valid = denominators > 0
    values = a[idx].sum(axis=1)[valid] / denominators[valid]
    return {
        "point": float(a.sum() / d.sum()),
        "ci95": [float(x) for x in np.percentile(values, [2.5, 97.5])],
        "denominator": int(d.sum()),
        "clusters": len(records),
        "replicates": 10000,
        "seed": 20261002,
        "unit": "canonical_input",
        "undefined_replicates": int((~valid).sum()),
        "estimator": "paired delta" if delta else "pooled accuracy",
    }


def semantic_metrics(items, gold, results):
    by_case = {(g["item_id"], g["requirement_id"], g["evidence_id"]): g["truth"] for g in gold}
    pair_records, status_records, clusters = [], [], []
    confusion = {g: {p: 0 for p in (*CLASSES, "MISSING_VERIFIER")} for g in CLASSES}
    for item in items:
        result = results.get(item["item_id"], {})
        available = result.get("semantic_result_available", False)
        predictions = (
            {
                (a["requirement_id"], a["evidence_id"]): a["verdict"]
                for a in result.get("assessments", [])
            }
            if available
            else {}
        )
        truths = {(r, e): by_case.get((item["item_id"], r, e)) for r, e in eligible_pairs(item)}
        cluster = {
            "item_id": item["item_id"],
            "n": 0,
            "exact": 0,
            "coverage": 0,
            "negative_n": 0,
            "false_support": 0,
            "status_n": 0,
            "status": 0,
        }
        for (rid, eid), truth in truths.items():
            if truth not in CLASSES:
                continue
            predicted = predictions.get((rid, eid), "MISSING_VERIFIER")
            exact = predicted == truth
            binary = predicted in CLASSES and ((predicted in POSITIVE) == (truth in POSITIVE))
            negative = truth not in POSITIVE
            false = negative and predicted in POSITIVE
            cluster["n"] += 1
            cluster["exact"] += exact
            cluster["coverage"] += binary
            cluster["negative_n"] += negative
            cluster["false_support"] += false
            confusion[truth][predicted] += 1
            pair_records.append(
                {
                    "item_id": item["item_id"],
                    "requirement_id": rid,
                    "evidence_id": eid,
                    "gold": truth,
                    "prediction": predicted,
                    "exact_correct": exact,
                    "coverage_correct": binary,
                    "false_support": false,
                    "scope_stratum": item["scope_stratum"],
                }
            )
        gs = gold_status(item, truths)
        ps = statuses(item, result["assessments"]) if available else {}
        for rid, truth in gs.items():
            predicted = ps.get(rid, "MISSING_VERIFIER")
            resolved = truth != "UNRESOLVED"
            if resolved:
                cluster["status_n"] += 1
                cluster["status"] += truth == predicted
            status_records.append(
                {
                    "item_id": item["item_id"],
                    "requirement_id": rid,
                    "gold": truth,
                    "prediction": predicted,
                    "resolved": resolved,
                    "correct": resolved and truth == predicted,
                }
            )
        clusters.append(cluster)
    classes = {}
    for label in CLASSES:
        n = sum(confusion[label].values())
        predicted_n = sum(confusion[g][label] for g in CLASSES)
        tp = confusion[label][label]
        recall = tp / n if n else None
        classes[label] = {
            "gold_n": n,
            "predicted_n": predicted_n,
            "precision": tp / predicted_n if predicted_n else None,
            "recall": recall,
            "gate": "NOT_POWERED" if n < 20 else ("PASS" if recall >= 0.75 else "FAIL"),
        }
    return {
        "coverage": clustered(clusters, "coverage"),
        "exact": clustered(clusters, "exact"),
        "status": clustered(clusters, "status", "status_n"),
        "false_support": clustered(clusters, "false_support", "negative_n"),
        "confusion": confusion,
        "classes": classes,
        "clusters": clusters,
        "pairs": pair_records,
        "requirements": status_records,
        "gold_unresolved_or_missing_pairs": sum(len(eligible_pairs(i)) for i in items)
        - len(pair_records),
    }


def qualification(n, gold_availability, reliability, metrics, intact=True):
    def accuracy(name, point, floor):
        v = metrics[name]
        return v["point"] is not None and v["point"] >= point and v["ci95"][0] >= floor

    checks = {
        "fresh_inputs": n >= 60,
        "gold_availability": gold_availability >= 0.90,
        "first_valid": reliability["first_valid_rate"] >= 0.98,
        "bounded_availability": reliability["bounded_availability"] >= 0.98,
        "truncation": reliability["truncation_rate"] <= 0.02,
        "scope_rejection": reliability["post_provider_scope_rejection"] == 0,
        "coverage_accuracy_and_ci": accuracy("coverage", 0.95, 0.90),
        "exact_accuracy_and_ci": accuracy("exact", 0.90, 0.80),
        "status_accuracy_and_ci": accuracy("status", 0.90, 0.80),
        "false_support": metrics["false_support"]["point"] is not None
        and metrics["false_support"]["point"] <= 0.05,
        "powered_class_recall": all(c["gate"] != "FAIL" for c in metrics["classes"].values()),
        "zero_leakage_drift_provenance_violation": intact,
    }
    return checks, all(checks.values())


def reliability(results):
    n = len(results)
    attempts = [a for r in results for a in r["attempts"]]
    diagnostics = Counter(d for a in attempts for d in a.get("pair_id_diagnostics", []))
    return {
        "inputs": n,
        "first_valid_rate": sum(r["first_attempt_valid"] for r in results) / n,
        "bounded_availability": sum(r["semantic_result_available"] for r in results) / n,
        "truncation_rate": sum(
            any(a["finish_reason"] == "length" for a in r["attempts"]) for r in results
        )
        / n,
        "actual_truncated_requests": sum(a["finish_reason"] == "length" for a in attempts),
        "actual_requests": len(attempts),
        "pair_id_failures": dict(diagnostics),
        "post_provider_scope_rejection": sum(
            "outside source scope" in (r.get("error") or "") for r in results
        ),
    }


def next_gold_attempt(attempts):
    return next_semantic_attempt(attempts)
