"""Zero-call construction audit and guarded identity utilities. No quality relaxation."""

import re
from collections import Counter

from paper_research.evaluation.phase2b import digest, normalize
from paper_research.evaluation.phase2c_r5 import overlap, span_map

LENGTH = "length_filter"
QUALITY = "clause_quality_filter"
NONPROSE = "table_reference_non_prose_filter"
REASONS = (
    "historical_source_anchor_identity",
    "historical_clause_identity",
    "historical_evidence_set_collision",
    "historical_canonical_input_collision",
    QUALITY,
    LENGTH,
    NONPROSE,
    "missing_provenance",
    "insufficient_candidate_evidence",
    "scope_construction_failure",
    "other",
)
GATES = (
    "length_90_to_420",
    "at_least_14_ascii_words",
    "frozen_scientific_predicate",
    "no_frozen_nonprose_marker",
)
VERB = (
    r"\b(is|are|can|has|have|uses|use|enables|improves|requires|allows|"
    r"show|find|results|achieves|propose|introduce)\b"
)
NONPROSE_MARKERS = (r"https?://", r"@", r"arXiv:", r"et al\.", r"References")


def audit_span(span):
    """Explain the exact four conjuncts in R5 clauses(); never use this to select new tasks."""
    raw = span["raw_text"]
    marker_hits = [p for p in NONPROSE_MARKERS if re.search(p, raw)]
    flags = (
        90 <= len(raw.strip()) <= 420,
        len(re.findall(r"[A-Za-z]{3,}", raw)) >= 14,
        bool(re.search(VERB, raw, re.I)),
        not marker_hits,
    )
    failed = next((i for i, v in enumerate(flags) if not v), None)
    return {
        "span_id": span["span_id"],
        "start": span["start"],
        "end": span["end"],
        "raw_text_sha256": digest(raw),
        "stripped_codepoints": len(raw.strip()),
        "ascii_words_at_least_3_letters": len(re.findall(r"[A-Za-z]{3,}", raw)),
        "predicate_checks": dict(zip(GATES, flags, strict=True)),
        "first_failed_predicate": GATES[failed] if failed is not None else None,
        "primary_reason": (
            LENGTH
            if failed == 0
            else QUALITY
            if failed in (1, 2)
            else NONPROSE
            if failed == 3
            else None
        ),
        "nonprose_marker_hits": marker_hits,
    }


def audit_chunk(chunk, historical_chunks, actual_clause_selector, known_papers):
    text = chunk["chunk_text"][:2400]
    source_ok = bool(
        chunk.get("block_ids")
        and chunk.get("page_start", 0) >= 1
        and chunk.get("paper_id") in known_papers
        and chunk.get("chunk_id")
    )
    spans = span_map(text)
    audits = [audit_span(s) for s in spans]
    cumulative = {
        key: sum(all(a["predicate_checks"][k] for k in GATES[: i + 1]) for a in audits)
        for i, key in enumerate(GATES)
    }
    original = actual_clause_selector(chunk)
    explained_ids = {a["span_id"] for a in audits if a["first_failed_predicate"] is None}
    assert explained_ids == {s["span_id"] for s in original}, "R5 predicate mirror drift"
    if not source_ok:
        primary = "missing_provenance"
    elif chunk["chunk_id"] in historical_chunks:
        primary = "historical_source_anchor_identity"
    elif not cumulative[GATES[0]]:
        primary = LENGTH
    elif not cumulative[GATES[2]]:
        primary = QUALITY
    elif not cumulative[GATES[3]]:
        primary = NONPROSE
    else:
        primary = None
    downstream = (
        "historical_clause_identity",
        "historical_evidence_set_collision",
        "historical_canonical_input_collision",
        "insufficient_candidate_evidence",
        "scope_construction_failure",
    )
    return {
        "unused_chunk_identity": chunk["chunk_id"],
        "paper_identity": chunk["paper_id"],
        "page_start": chunk["page_start"],
        "raw_prefix_sha256": digest(text),
        "source_provenance_available": source_ok,
        "historical_whole_chunk_collision": chunk["chunk_id"] in historical_chunks,
        "raw_span_count": len(spans),
        "span_audit": audits,
        "cumulative_surviving_span_counts": cumulative,
        "r5_selector_clause_count": len(original),
        "primary_exclusion_reason": primary,
        "reason_labels": sorted({a["primary_reason"] for a in audits if a["primary_reason"]}),
        "downstream_checks": {
            r: {"status": "NOT_REACHED_NO_QUALIFIED_CLAUSE", "evaluated": False} for r in downstream
        }
        if not original
        else {},
        "standalone_table_or_title_filter": "NOT_PRESENT_IN_FROZEN_R5_SELECTOR",
        "table_like_text_can_fail_existing_length_word_or_predicate_rules": True,
    }


def exclusion_funnel(audits):
    chunks = {g: sum(a["cumulative_surviving_span_counts"][g] > 0 for a in audits) for g in GATES}
    spans = {g: sum(a["cumulative_surviving_span_counts"][g] for a in audits) for g in GATES}
    primary = Counter(a["primary_exclusion_reason"] for a in audits)
    span_failures = Counter(s["first_failed_predicate"] for a in audits for s in a["span_audit"])
    qualified = sum(a["r5_selector_clause_count"] for a in audits)
    coarse = sum(
        a["historical_whole_chunk_collision"] and a["r5_selector_clause_count"] > 0 for a in audits
    )
    return {
        "initial_untouched_usable_chunks": len(audits),
        "source_provenance_pass": sum(a["source_provenance_available"] for a in audits),
        "whole_chunk_identity_pass": sum(not a["historical_whole_chunk_collision"] for a in audits),
        "deterministic_span_extraction_chunks": sum(a["raw_span_count"] > 0 for a in audits),
        "raw_spans": sum(a["raw_span_count"] for a in audits),
        "cumulative_quality_chunk_survivors": chunks,
        "cumulative_quality_span_survivors": spans,
        "qualified_scientific_clause_count": qualified,
        "coarse_identity_excluded_qualified_chunks": coarse,
        "primary_chunk_reason_counts": {r: primary[r] for r in REASONS},
        "primary_span_predicate_failure_counts": dict(span_failures),
        "historical_clause_freshness": {"status": "NOT_REACHED", "evaluated_clauses": 0}
        if not qualified
        else {"status": "PENDING"},
        "candidate_construction": {"status": "NOT_REACHED", "inputs": 0}
        if not qualified
        else {"status": "PENDING"},
        "final_inputs": 0 if not qualified else None,
        "conclusion": "NO_QUALIFIED_CLAUSE_UNDER_FROZEN_R5_RULES"
        if not qualified
        else "FRESHNESS_GRANULARITY_BLOCKAGE_PROVEN"
        if coarse
        else "FURTHER_AUDIT_REQUIRED",
    }


def require_clause_level_recovery(funnel):
    if not (
        funnel["conclusion"] == "FRESHNESS_GRANULARITY_BLOCKAGE_PROVEN"
        and funnel["qualified_scientific_clause_count"] > 0
        and funnel["coarse_identity_excluded_qualified_chunks"] > 0
    ):
        raise ValueError("CLAUSE_LEVEL_FRESHNESS_NOT_AUTHORIZED_BY_AUDIT")


def clause_identity(raw_text, paper_id, provenance):
    """Identity normalization only. Caller retains untouched source and requirement text."""
    return {
        "exact_clause_hash": digest(raw_text),
        "normalized_clause_hash": digest(normalize(raw_text)),
        "scientific_source_clause_hash": digest([normalize(raw_text), paper_id, provenance]),
    }


def historical_clause_guard(raw, paper, provenance, historical):
    ident = clause_identity(raw, paper, provenance)
    return [key for key, value in ident.items() if value in historical.get(key, set())]


def overlap_guard(payload, anchors, prior, development_chunks):
    reasons = overlap(payload, anchors, prior)
    exposed = {e["chunk_id"] for e in payload["evidence"]} | {a["chunk_id"] for a in anchors}
    if exposed & development_chunks:
        reasons.append("DEVELOPMENT_EXAMPLE_EXPOSURE")
    return reasons


def deterministic_selection(pool, funnel, target=60):
    """Unavailable in this phase unless the audit itself establishes the authorized mechanism."""
    require_clause_level_recovery(funnel)
    if target != 60:
        raise ValueError("FIXED_TARGET_60")
    ordered = sorted(
        pool,
        key=lambda x: (
            x["paper_id"],
            x["source_position"],
            x["clause_position"],
            x["canonical_input_hash"],
        ),
    )
    used_chunks, paper_count, selected, excluded = set(), Counter(), [], []
    for item in ordered:
        anchors = item["source_clauses"]
        chunks = [a["chunk_id"] for a in anchors]
        reason = (
            "PAPER_LIMIT_2"
            if paper_count[item["paper_id"]] >= 2
            else "SOURCE_CHUNK_LIMIT_1"
            if len(chunks) != len(set(chunks)) or set(chunks) & used_chunks
            else "TARGET_ALREADY_REACHED"
            if len(selected) == target
            else None
        )
        if reason:
            excluded.append(
                {"canonical_input_hash": item["canonical_input_hash"], "reason": reason}
            )
            continue
        selected.append(item)
        paper_count[item["paper_id"]] += 1
        used_chunks.update(chunks)
    return selected, excluded


def conditional_formal_guard(manifest, *, arm, candidate_hash_unchanged, gold=None):
    if (
        manifest["n"] < 60
        or not manifest.get("benchmark_frozen")
        or not manifest.get("clause_level_recovery_authorized")
    ):
        raise ValueError("NO_PROVIDER_BEFORE_VALID_FRESH_BENCHMARK_FREEZE")
    if not candidate_hash_unchanged:
        raise ValueError("CANDIDATE_IDENTITY_DRIFT")
    if arm not in ("gold", "v1_2", "v1_3"):
        raise ValueError("UNKNOWN_FORMAL_ARM")
    if arm != "gold" and not (gold and gold.get("frozen") and gold.get("availability", 0) >= 0.9):
        raise ValueError("GOLD_BEFORE_VERIFIER")
