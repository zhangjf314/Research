# ruff: noqa: E501
"""Build the entirely offline Stage 13.4 Phase A review and coverage artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from paper_research.evaluation.canonical_hash import hash_with_metadata
from paper_research.generation.citation_registry import CitationRegistry

try:
    from scripts.evidence_qa_dev_lib_v1 import (
        DATA,
        DEV_IDS,
        DOCS,
        canonical_hash,
        overlap,
        read_jsonl,
        terms,
    )
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import (  # type: ignore[no-redef]
        DATA,
        DEV_IDS,
        DOCS,
        canonical_hash,
        overlap,
        read_jsonl,
        terms,
    )

ROOT = DATA.parents[1]
RUN_ROOT = DATA / "evidence-qa-dev-v2/runs"
AUDIT = DATA / "evidence-qa-dev-v2-citation-audit-v1.jsonl"
COVERAGE_JSONL = DATA / "dev-v2-claim-coverage-audit-v1.jsonl"
COVERAGE_CSV = DATA / "dev-v2-claim-coverage-audit-v1.csv"
COVERAGE_DOC = DOCS / "dev-v2-claim-coverage-audit-v1.md"
COUNTERFACTUAL_JSON = DATA / "dev-v2-claim-coverage-counterfactual-v1.json"
COUNTERFACTUAL_DOC = DOCS / "dev-v2-claim-coverage-counterfactual-v1.md"
GUIDE = DOCS / "evidence-qa-dev-v2-citation-review-guide-v1.md"
PROMPT_DOC = DOCS / "qa-required-claims-citation-id-v3.md"
PACK = ROOT / "artifacts/stage13-4-dev-v2-citation-review-pack.zip"
EVIDENCE_PATH = DATA / "evidence-corpus-v1.jsonl"
ALLOWED_LABELS = {"fully_supported", "partially_supported", "related_but_insufficient", "unsupported", "gold_annotation_too_narrow", "ambiguous_claim", "malformed_evidence"}
HUMAN_FIELDS = {"human_review_status", "human_label", "reviewer", "reviewed_at", "review_notes"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def run_data() -> dict[str, dict[str, Any]]:
    output = {}
    for path in RUN_ROOT.glob("*/result.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        run_dir = path.parent
        output[result["question_id"]] = {"result": result, "registry": CitationRegistry.model_validate_json((run_dir / "citation-registry.json").read_text(encoding="utf-8")), "context": json.loads((run_dir / "context-trace.json").read_text(encoding="utf-8")), "retrieval": json.loads((run_dir / "retrieval-trace.json").read_text(encoding="utf-8")), "raw": json.loads((run_dir / "raw-provider-response.json").read_text(encoding="utf-8"))}
    if set(output) != set(DEV_IDS):
        raise RuntimeError("Dev v2 run set is not the frozen ten-question manifest")
    return output


def enrich_audit(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = read_jsonl(AUDIT)
    if len(rows) != 57 or len({row["sample_id"] for row in rows}) != 57:
        raise RuntimeError("citation audit must contain 57 unique samples")
    evidence_rows = read_jsonl(EVIDENCE_PATH)
    evidence = {(row["paper_id"], int(row["page"]), row["block_id"]): row for row in evidence_rows}
    by_block = {(row["paper_id"], row["block_id"]): row for row in evidence_rows}
    gold = {row["question_id"]: row for row in read_jsonl(DATA / "gold-set-v1.jsonl")}
    claims = defaultdict(list)
    for row in read_jsonl(DATA / "claim-units-v1.jsonl"):
        claims[row["question_id"]].append(row)
    corpus_metadata = hash_with_metadata(EVIDENCE_PATH, "canonical_jsonl_v1")
    enriched = []
    for old in rows:
        row = json.loads(json.dumps(old))
        qid = row["question_id"]
        run = runs[qid]
        triple = (row["citation"]["paper_id"], int(row["citation"]["page"]), row["citation"]["block_id"])
        unit = evidence.get(triple)
        if unit is None:
            raise RuntimeError(f"missing citation triple for {row['sample_id']}")
        entry = next((item for item in run["registry"].entries if item.triple == triple), None)
        if entry is None:
            raise RuntimeError(f"citation triple absent from registry: {row['sample_id']}")
        previous = by_block.get((unit["paper_id"], unit.get("previous_block_id")))
        following = by_block.get((unit["paper_id"], unit.get("next_block_id")))
        adjacent = {item["block_id"] for item in run["context"].get("adjacent_completion_blocks", [])}
        matches = sorted(({"required_claim_id": item["claim_id"], "required_claim_text": item["claim_text"], "token_overlap": overlap(item["claim_text"], row["claim_text"])} for item in claims[qid]), key=lambda item: item["token_overlap"], reverse=True)
        row.update({"variant": "citation_id_v2_adjacent_same_page_completion", "question": gold[qid]["question"], "answerable": gold[qid]["answerable"], "required_claim_match": {"best": matches[0] if matches else None, "all": matches}, "citation_id": entry.citation_id, "citation_triple": {"paper_id": triple[0], "page": triple[1], "block_id": triple[2]}, "cited_evidence_context": {"previous": {"block_id": previous["block_id"], "text": previous["text"]} if previous else None, "current": {"block_id": unit["block_id"], "text": unit["text"]}, "next": {"block_id": following["block_id"], "text": following["text"]} if following else None}, "evidence_source": "adjacent_completion" if unit["block_id"] in adjacent else "original_selected", "block_type": unit["block_type"], "semantic_token_signal": round(len(terms(row["claim_text"]) & terms(unit["text"])) / max(1, len(terms(row["claim_text"]))), 6), "registry_hash": run["registry"].registry_hash, "source_hash": corpus_metadata["raw_value_at_review"], "source_canonical_sha256": corpus_metadata["value"], "source_hash_mode": corpus_metadata["mode"], "source_hash_schema_version": corpus_metadata["schema_version"], "source_raw_sha256_at_review": corpus_metadata["raw_value_at_review"], "source_legacy_raw_hash_verified_via_newline_normalization": False, "source_record_hash": canonical_hash(unit)})
        immutable = {key: value for key, value in row.items() if key not in HUMAN_FIELDS | {"immutable_record_hash"}}
        row["immutable_record_hash"] = canonical_hash(immutable)
        enriched.append(row)
    write_jsonl(AUDIT, enriched)
    return enriched


def generated_claims(result: dict[str, Any]) -> list[dict[str, Any]]:
    return result.get("answer", {}).get("claims", []) if result["status"] == "completed" else []


def coverage_audit(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    gold = {row["question_id"]: row for row in read_jsonl(DATA / "gold-set-v1.jsonl")}
    claim_units = defaultdict(list)
    for row in read_jsonl(DATA / "claim-units-v1.jsonl"):
        if row["question_id"] in DEV_IDS and row.get("expected_answerability"):
            claim_units[row["question_id"]].append(row)
    rows = []
    for qid in DEV_IDS:
        run = runs[qid]
        result = run["result"]
        registry = run["registry"]
        final_blocks = {entry.block_id for entry in registry.entries}
        final_triples = [list(entry.triple) for entry in registry.entries]
        generated = generated_claims(result)
        scores_by_generated = {item.get("claim_id", f"g{index}"): [overlap(unit["claim_text"], item.get("claim_text", "")) for unit in claim_units[qid]] for index, item in enumerate(generated)}
        for unit in claim_units[qid]:
            ranked = sorted(((overlap(unit["claim_text"], item.get("claim_text", "")), item) for item in generated), key=lambda pair: pair[0], reverse=True)
            best_score, best = ranked[0] if ranked else (0.0, None)
            current_covered = best_score >= 0.35
            relaxed = best_score >= 0.25
            merged = bool(not current_covered and relaxed and best and sum(score >= 0.25 for score in scores_by_generated.get(best.get("claim_id", ""), [])) > 1)
            gold_in_context = bool(set(unit["gold_block_ids"]) & final_blocks)
            if result["status"] == "validation_failed":
                stage = "malformed_json" if "JSONDecodeError" in (result.get("failure_reason") or "") else "schema_validation_failure"
            elif current_covered:
                stage = None
            elif not gold_in_context:
                stage = "retrieval_ranked_out"
            elif relaxed:
                stage = "required_claim_matching_failure"
            else:
                stage = "model_omitted_claim"
            citations = best.get("citations", []) if best else []
            citation_ids = best.get("citation_ids", []) if best else []
            rows.append({"question_id": qid, "question": gold[qid]["question"], "required_claim_id": unit["claim_id"], "required_claim_text": unit["claim_text"], "claim_role": unit["claim_role"], "target_paper_ids": unit["target_paper_ids"], "gold_block_ids": unit["gold_block_ids"], "gold_pages": unit["gold_pages"], "retrieval_candidate_evidence": {"materialized": False, "candidate_count": run["retrieval"]["candidate_count"], "note": "Dev v2 trace persists candidate count but not the complete candidate triples."}, "final_context_evidence": {"triples": final_triples, "gold_block_available": gold_in_context}, "claim_specific_allocated_evidence": [], "evidence_completeness_before_generation": gold_in_context, "prompt_claim_representation": {"required_claim_explicitly_present": False, "reason": "qa-production-citation-id-v2 payload contains question/evidence/registry but not required_claims"}, "model_output_claim": best, "generated": current_covered, "omitted": not current_covered and not merged, "merged_into_other_claim": merged, "contradicted": False, "unsupported_before_generation": not gold_in_context, "unsupported_after_generation": bool(best and not any(citation["block_id"] in unit["gold_block_ids"] for citation in citations)), "citation_ids": citation_ids, "citation_triples": citations, "citation_support_human_label": None, "coverage_credit": 1 if current_covered else 0, "coverage_failure_stage": stage, "automatic_failure_reason": None if current_covered else f"best deterministic token overlap={best_score:.6f}; gold_in_context={gold_in_context}; run_status={result['status']}", "human_review_status": "pending", "reviewer": None, "reviewed_at": None, "review_notes": None, "matcher_scores": {"exact_normalized": int(bool(best) and re.sub(r'\W+', ' ', unit['claim_text'].lower()).strip() == re.sub(r'\W+', ' ', best.get('claim_text', '').lower()).strip()), "token_overlap": best_score, "lexical_entailment_candidate": relaxed}, "run_id": result["run_id"], "run_status": result["status"], "schema_version": "dev-v2-claim-coverage-audit-v1"})
    write_jsonl(COVERAGE_JSONL, rows)
    flat = [{key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()} for row in rows]
    with COVERAGE_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    return rows


def context_variant(registry: CitationRegistry, adjacent: set[str], gold_blocks: set[str], mode: str) -> dict[str, Any]:
    evidence = {(row["paper_id"], row["block_id"]): row for row in read_jsonl(EVIDENCE_PATH)}
    entries = registry.entries
    if mode == "remove_weak_adjacent":
        entries = [entry for entry in entries if entry.block_id not in adjacent]
    elif mode == "cap_adjacent_25pct":
        originals = [entry for entry in entries if entry.block_id not in adjacent]
        extras = [entry for entry in entries if entry.block_id in adjacent]
        original_tokens = sum(len(evidence[(entry.paper_id, entry.block_id)]["text"].split()) for entry in originals)
        kept, adjacent_tokens = [], 0
        for entry in extras:
            tokens = len(evidence[(entry.paper_id, entry.block_id)]["text"].split())
            if adjacent_tokens + tokens <= max(1, original_tokens // 3):
                kept.append(entry)
                adjacent_tokens += tokens
        entries = originals + kept
    tokens = sum(len(evidence[(entry.paper_id, entry.block_id)]["text"].split()) for entry in entries)
    blocks = {entry.block_id for entry in entries}
    return {"selected_blocks": len(entries), "token_count": tokens, "gold_blocks_available": len(blocks & gold_blocks), "gold_block_recall": len(blocks & gold_blocks) / max(1, len(gold_blocks)), "adjacent_blocks": sum(entry.block_id in adjacent for entry in entries)}


def counterfactual(runs: dict[str, dict[str, Any]], coverage: list[dict[str, Any]]) -> dict[str, Any]:
    exact = sum(row["matcher_scores"]["exact_normalized"] for row in coverage)
    current = sum(row["coverage_credit"] for row in coverage)
    lexical = sum(row["matcher_scores"]["lexical_entailment_candidate"] for row in coverage)
    false_positive_candidates = [row["required_claim_id"] for row in coverage if 0.25 <= row["matcher_scores"]["token_overlap"] < 0.35]
    parser = {"valid_raw_responses": 0, "strict_failures": [], "claims_preserved": 0, "diagnostic_salvage_only": []}
    context = {}
    gold = {row["question_id"]: row for row in read_jsonl(DATA / "gold-set-v1.jsonl")}
    for qid, run in runs.items():
        raw_content = run["raw"]["choices"][0]["message"]["content"]
        try:
            decoded = json.loads(raw_content)
            parser["valid_raw_responses"] += 1
            parser["claims_preserved"] += len(decoded.get("claims", []))
        except json.JSONDecodeError as exc:
            parser["strict_failures"].append({"question_id": qid, "error": f"{type(exc).__name__}: {exc}"})
            parser["diagnostic_salvage_only"].append(qid)
        adjacent = {item["block_id"] for item in run["context"].get("adjacent_completion_blocks", [])}
        gold_blocks = set(gold[qid]["gold_block_ids"])
        context[qid] = {mode: context_variant(run["registry"], adjacent, gold_blocks, mode) for mode in ("current", "remove_weak_adjacent", "cap_adjacent_25pct")}
    stages = Counter(row["coverage_failure_stage"] or "covered" for row in coverage)
    by_question = {qid: {"required": sum(row["question_id"] == qid for row in coverage), "covered": sum(row["question_id"] == qid and row["coverage_credit"] for row in coverage), "omitted": sum(row["question_id"] == qid and row["omitted"] for row in coverage), "merged": sum(row["question_id"] == qid and row["merged_into_other_claim"] for row in coverage)} for qid in DEV_IDS}
    return {"schema_version": "dev-v2-claim-coverage-counterfactual-v1", "live_llm_calls": 0, "embedding_api_calls": 0, "historical_runs_modified": False, "parser_replay": parser, "matcher_replay": {"required_claim_count": len(coverage), "original_coverage": current / max(1, len(coverage)), "exact_normalized_coverage": exact / max(1, len(coverage)), "token_overlap_0_35_coverage": current / max(1, len(coverage)), "lexical_entailment_0_25_candidate_coverage": lexical / max(1, len(coverage)), "one_generated_claim_to_multiple_required_claims_candidate_coverage": lexical / max(1, len(coverage)), "new_candidate_matches": lexical - current, "possible_false_positive_claim_ids": false_positive_candidates, "human_adjudication_required": bool(false_positive_candidates)}, "context_budget_replay": context, "coverage_failure_stage_distribution": dict(sorted(stages.items())), "per_question": by_question, "prompt_v2_audit": {"required_claims_explicitly_enumerated": False, "claim_evidence_one_to_one": False, "model_infers_claims": True, "silent_omission_risk": True, "max_output_tokens": 2048, "short_answer_bias_risk": True}, "fix_candidates": [{"name": "required_claim_checklist_and_slots", "hypothesis": "Explicit slots prevent silent omission.", "observed_evidence": "v2 payload omits required claims; omitted claims exist despite available Gold blocks.", "affected_questions": [row["question_id"] for row in coverage if row["omitted"]], "expected_benefit": "coverage omissions become explicit answered/unsupported outcomes", "citation_risk": "low when IDs are claim-allocated", "latency_token_impact": "moderate output increase proportional to required claim count", "implementation_complexity": "medium", "rollback": "retain citation-id-v2", "whether_requires_new_llm_run": True}, {"name": "deterministic_matcher_adjudication", "hypothesis": "Current 0.35 token threshold causes false negatives for paraphrases.", "observed_evidence": false_positive_candidates, "affected_questions": sorted({row["question_id"] for row in coverage if row["required_claim_id"] in false_positive_candidates}), "expected_benefit": "recover diagnostic coverage without changing model output", "citation_risk": "medium; requires human adjudication", "latency_token_impact": "none", "implementation_complexity": "low", "rollback": "restore 0.35 matcher", "whether_requires_new_llm_run": False}, {"name": "adjacent_token_cap_and_original_priority", "hypothesis": "Cap weak adjacent evidence while retaining original evidence.", "observed_evidence": "offline context token variants only; no model-quality claim", "affected_questions": DEV_IDS, "expected_benefit": "lower context tokens with measured Gold availability", "citation_risk": "must reject variants that reduce Gold recall", "latency_token_impact": "reduced input tokens", "implementation_complexity": "low", "rollback": "current adjacent completion", "whether_requires_new_llm_run": True}]}


def write_docs_and_pack(audit: list[dict[str, Any]], coverage: list[dict[str, Any]], cf: dict[str, Any]) -> None:
    GUIDE.write_text("# Dev v2 Citation Review Guide\n\nReview all 57 samples against the claim and cited evidence. Labels: `fully_supported`, `partially_supported`, `related_but_insufficient`, `unsupported`, `gold_annotation_too_narrow`, `ambiguous_claim`, `malformed_evidence`. Do not infer a label from automated signals. Reviewer and notes are mandatory. Verify the exact citation ID against the run registry and the paper/page/block triple before approval. This AI-assisted review pack is not an independent double-blind human audit and cannot be extrapolated to Full-50.\n", encoding="utf-8")
    stages = Counter(row["coverage_failure_stage"] or "covered" for row in coverage)
    per_q = cf["per_question"]
    lines = ["# Dev v2 Required Claim Coverage Audit", "", f"- Required claims: {len(coverage)}", f"- Covered: {sum(row['coverage_credit'] for row in coverage)}", f"- Omitted: {sum(row['omitted'] for row in coverage)}", f"- Merged candidates: {sum(row['merged_into_other_claim'] for row in coverage)}", f"- Unsupported before generation: {sum(row['unsupported_before_generation'] for row in coverage)}", "", "## Failure stages", ""] + [f"- {key}: {value}" for key, value in sorted(stages.items())] + ["", "## Per question", "", "| Question | Required | Covered | Omitted | Merged |", "|---|---:|---:|---:|---:|"] + [f"| {qid} | {per_q[qid]['required']} | {per_q[qid]['covered']} | {per_q[qid]['omitted']} | {per_q[qid]['merged']} |" for qid in DEV_IDS] + ["", "Required claims were not explicitly represented in the v2 prompt payload. Candidate triples were not fully materialized in the historical trace, so candidate-missing versus ranked-out cannot always be separated; this limitation is preserved rather than inferred."]
    COVERAGE_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    COUNTERFACTUAL_DOC.write_text("# Dev v2 Claim Coverage Counterfactual\n\n" f"- Parser replay: {cf['parser_replay']['valid_raw_responses']}/10 strict JSON valid; q050 remains a strict failure and diagnostic salvage does not replace the formal result.\n- Original matcher coverage: {cf['matcher_replay']['original_coverage']:.6f}\n- Lexical 0.25 candidate coverage: {cf['matcher_replay']['lexical_entailment_0_25_candidate_coverage']:.6f}\n- Possible false-positive matches requiring human adjudication: {len(cf['matcher_replay']['possible_false_positive_claim_ids'])}\n- Context replay is availability/token-only and makes no model-quality claim.\n- Live LLM calls: 0; Embedding API calls: 0.\n", encoding="utf-8")
    PROMPT_DOC.write_text("# qa-required-claims-citation-id-v3 (offline draft)\n\nEach input required claim has `required_claim_id`, `claim_text`, `evidence_complete`, and claim-local `allowed_citation_ids`. Every answerable output must contain exactly one slot per required claim with `status=answered|unsupported|not_applicable`, `claim_text`, `citation_ids`, and `omission_reason`. Answered slots require claim-local citations; unsupported/not-applicable slots forbid citations and require a reason. Unknown IDs, cross-claim borrowing, missing slots, and malformed JSON fail strictly. Unanswerable output requires `answerable=false`, `claims=[]`, and a refusal reason. Output budget should scale deterministically with required-claim count; no live run is authorized in Stage 13.4 Phase A.\n", encoding="utf-8")
    members = [AUDIT, EVIDENCE_PATH, DATA / "claim-units-v1.jsonl", DATA / "gold-set-v1.jsonl", DATA / "retrieval-gold-v2.jsonl", DATA / "evidence-qa-dev-v2.json", DOCS / "evidence-qa-dev-v2.md", GUIDE]
    PACK.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PACK, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in members:
            info = zipfile.ZipInfo(path.name, date_time=(2026, 7, 15, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    with zipfile.ZipFile(PACK) as archive:
        names = archive.namelist()
        if any(name.lower().startswith(".env") or name.lower().endswith((".sqlite", ".db")) for name in names):
            raise RuntimeError("unsafe review pack member")
        body = b"".join(archive.read(name) for name in names)
        if b'"Authorization":' in body or b"Bearer sk-" in body:
            raise RuntimeError("authorization material found in review pack")


def main() -> None:
    runs = run_data()
    audit = enrich_audit(runs)
    coverage = coverage_audit(runs)
    cf = counterfactual(runs, coverage)
    COUNTERFACTUAL_JSON.write_text(json.dumps(cf, ensure_ascii=False, indent=2), encoding="utf-8")
    write_docs_and_pack(audit, coverage, cf)
    print(json.dumps({"citation_samples": len(audit), "pending": sum(row["human_review_status"] == "pending" for row in audit), "required_claims": len(coverage), "covered": sum(row["coverage_credit"] for row in coverage), "pack": str(PACK), "pack_sha256": sha256(PACK), "live_llm_calls": 0}))


if __name__ == "__main__":
    main()
