"""R6B diagnostics. Absolute frozen R5 gates remain separate from mechanism evidence."""

from collections import Counter

from paper_research.evaluation.phase2c_r5 import (
    GATES,
    POSITIVE,
    clustered,
    qualification,
    reliability,
)


def mechanism_comparison(baseline_metrics, candidate_metrics):
    e0 = {r["item_id"]: r for r in baseline_metrics["clusters"]}
    e1 = {r["item_id"]: r for r in candidate_metrics["clusters"]}
    if set(e0) != set(e1):
        raise ValueError("UNPAIRED_CANONICAL_INPUTS")
    paired = []
    for iid, b in e0.items():
        c = e1[iid]
        if b["negative_n"] != c["negative_n"]:
            raise ValueError("UNPAIRED_NEGATIVE_GOLD_DENOMINATOR")
        paired.append(
            {
                "item_id": iid,
                "delta": c["false_support"] - b["false_support"],
                "negative_n": b["negative_n"],
            }
        )
    delta = clustered(paired, "delta", "negative_n", delta=True)
    if delta["point"] is None:
        mechanism = "NOT_EVALUATED"
    elif delta["point"] >= 0:
        mechanism = "MECHANISM_NOT_SUPPORTED"
    elif delta["ci95"][1] < 0:
        mechanism = "GROUNDED_POSITIVE_SUPPORT_MECHANISM_SUPPORTED"
    else:
        mechanism = "DIRECTIONAL_ONLY"
    return {
        "false_support_e1_minus_e0": delta,
        "mechanism": mechanism,
        "absolute_qualification_independent": True,
    }


def candidate_reliability(results):
    base = reliability(results)
    n = len(results)
    first = [r.get("grounding_attempts", [])[:1] for r in results]
    used = [
        {"item_id": r["item_id"], "attempt": a["attempt"]}
        for r in results
        for a in r.get("grounding_attempts", [])
        if a["singleton_canonicalized"]
    ]
    accepted = [a for r in results for a in r.get("grounding_attempts", []) if a["accepted_result"]]
    base.update(
        raw_first_schema_valid=sum(bool(a and a[0]["raw_schema_valid"]) for a in first) / n
        if n
        else None,
        accepted_first_result=base["first_valid_rate"],
        singleton_canonicalization_count=len(used),
        singleton_input_rate=len({a["item_id"] for a in used}) / n if n else None,
        singleton_attempts=used,
        grounding_integrity=all(a["grounding_integrity"] for a in accepted),
        accepted_grounding_records=len(accepted),
    )
    return base


def candidate_result_record(item_id, assessments, diagnostics, error=None):
    """Serializable phase-local result boundary; raw responses remain independently immutable."""
    import copy

    attempts = copy.deepcopy(diagnostics)
    accepted = [a for a in attempts if a["accepted_result"]]
    if len(accepted) > 1 or (error is not None and accepted):
        raise ValueError("INEXACT_ACCEPTED_RESULT")
    if accepted and attempts[-1] is not accepted[0]:
        raise ValueError("REQUEST_AFTER_VALID_RESULT")
    return {
        "item_id": item_id,
        "first_attempt_valid": bool(attempts and attempts[0]["accepted_result"]),
        "raw_first_schema_valid": bool(attempts and attempts[0]["raw_schema_valid"]),
        "semantic_result_available": bool(accepted) and error is None,
        "error": error,
        "assessments": [a.model_dump(mode="json") for a in assessments] if accepted else [],
        "grounding_attempts": attempts,
        "attempts": [
            {
                "valid": a["accepted_result"],
                "finish_reason": a.get("finish_reason"),
                "singleton_canonicalized": a["singleton_canonicalized"],
                "raw_schema_valid": a["raw_schema_valid"],
            }
            for a in attempts
        ],
        "singleton_canonicalized": any(a["singleton_canonicalized"] for a in attempts),
    }


def qualify_candidate(n, gold_availability, engineering, metrics, intact=True):
    checks, _ = qualification(n, gold_availability, engineering, metrics, intact)
    checks["grounding_integrity"] = bool(engineering["grounding_integrity"])
    return checks, all(checks.values())


def error_families(metrics):
    conf = metrics["confusion"]
    return {
        "IRRELEVANT_to_PARTIAL": conf["IRRELEVANT"]["PARTIAL"],
        "IRRELEVANT_gold_n": sum(conf["IRRELEVANT"].values()),
        "IRRELEVANT_to_PARTIAL_rate": conf["IRRELEVANT"]["PARTIAL"]
        / sum(conf["IRRELEVANT"].values())
        if sum(conf["IRRELEVANT"].values())
        else None,
        "SUPPORTS_to_PARTIAL": conf["SUPPORTS"]["PARTIAL"],
        "PARTIAL_to_SUPPORTS": conf["PARTIAL"]["SUPPORTS"],
    }


def grounding_diagnostics(results, gold):
    """Overlap is descriptive, never a gate or an entailment judgment."""
    import statistics

    truth = {(g["item_id"], g["requirement_id"], g["evidence_id"]): g for g in gold}
    counts = {v: [] for v in ("SUPPORTS", "PARTIAL", "CONTRADICTS")}
    overlaps, false = [], []
    for result in results:
        if not result["semantic_result_available"]:
            continue
        accepted = [a for a in result["grounding_attempts"] if a["accepted_result"]]
        if len(accepted) != 1:
            raise ValueError("INEXACT_ACCEPTED_GROUNDING_RESULT")
        for p in accepted[0]["grounding"]:
            key = (result["item_id"], p["requirement_id"], p["evidence_id"])
            g = truth.get(key)
            if p["verdict"] in counts:
                counts[p["verdict"]].append(len(p["support_span_ids"]))
            if g is None or g["truth"] == "UNRESOLVED":
                continue
            overlap = sorted(set(p["support_span_ids"]) & set(g["support_span_ids"]))
            overlaps.append(
                {
                    "item_id": key[0],
                    "requirement_id": key[1],
                    "evidence_id": key[2],
                    "span_intersection": overlap,
                }
            )
            if g["truth"] not in POSITIVE and p["verdict"] in POSITIVE:
                false.append(
                    {
                        "item_id": key[0],
                        **p,
                        "gold_truth": g["truth"],
                        "gold_support_span_ids": g["support_span_ids"],
                        "format_valid_but_positive_disagrees_with_frozen_gold": True,
                        "independent_science_rejudgment": False,
                    }
                )
    pooled = [n for ns in counts.values() for n in ns]
    counts["all_positive_or_contradictory"] = pooled
    counts["positive_support_only"] = [n for v in POSITIVE for n in counts[v]]
    return {
        "span_use": {
            v: {
                "n": len(ns),
                "mean": statistics.mean(ns) if ns else None,
                "median": statistics.median(ns) if ns else None,
            }
            for v, ns in counts.items()
        },
        "gold_span_overlap": overlaps,
        "overlap_is_not_a_gate": True,
        "false_support_forensic": false,
        "false_support_truth_counts": dict(Counter(p["gold_truth"] for p in false)),
        "frozen_gates": GATES,
    }
