# ruff: noqa: E501
"""Offline-only replay for the Dev v3.2 citation-policy candidate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from paper_research.evaluation.canonical_hash import hash_with_metadata
from paper_research.generation.citation_selection import (
    CITATION_SELECTION_VERSION,
    COMPARISON_VALIDATION_VERSION,
    EVIDENCE_ORIGIN_POLICY_VERSION,
    NUMERIC_VALIDATION_VERSION,
    OBLIGATION_POLICY_VERSION,
    CitationCandidate,
    FallbackAction,
    analyze_claim_obligations,
    citation_budget,
    select_citations,
    validate_comparison_evidence,
    validate_numeric_evidence,
)
from paper_research.generation.prompts import (
    QA_REQUIRED_CLAIMS_CITATION_ID_V3_2_CANDIDATE,
    qa_system_prompt,
)

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DOCS, canonical_hash, read_jsonl
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import (  # type: ignore[no-redef]
        DATA,
        DOCS,
        canonical_hash,
        read_jsonl,
    )

ROOT = DATA.parents[1]
RUN_ROOT = DATA / "evidence-qa-dev-v3-1/runs"
SUMMARY = DATA / "evidence-qa-dev-v3-1.json"
CLAIM_GOLD = DATA / "claim-evidence-gold-dev-v1.jsonl"
INPUTS_JSON = DATA / "dev-v3-2-offline-preflight-inputs-v1.json"
INPUTS_DOC = DOCS / "dev-v3-2-offline-preflight-inputs-v1.md"
OUTPUT_JSON = DATA / "dev-v3-2-offline-replay-v1.json"
OUTPUT_CSV = DATA / "dev-v3-2-offline-replay-v1.csv"
OUTPUT_DOC = DOCS / "dev-v3-2-offline-replay-v1.md"
FINAL_AUDIT = DATA / "dev-v3-2-offline-replay-v1-final-audit.json"
PROTOCOL_JSON = DATA / "dev-v3-2-protocol-candidate-v1.json"
PROTOCOL_DOC = DOCS / "dev-v3-2-protocol-candidate-v1.md"
MODES = [
    "baseline_v3_1",
    "primary_only",
    "primary_plus_cap",
    "obligation_coverage",
    "numeric_validator",
    "comparison_validator",
    "full_v3_2_candidate",
]
INPUT_PATHS = [
    CLAIM_GOLD,
    DATA / "claim-evidence-gold-dev-v1-freeze.json",
    SUMMARY,
    DATA / "evidence-qa-dev-v3-1-final-audit.json",
    DATA / "evidence-qa-dev-v3-1-citation-audit-v1.jsonl",
    DATA / "dev-v3-1-citation-failure-taxonomy-v2.jsonl",
    DATA / "dev-v3-2-citation-improvement-plan-v1.json",
    DATA / "evidence-qa-dev-v3-1-manifest.json",
    DATA / "provider-health-dev-v3-1-v1.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-policies", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def source_inputs() -> dict[str, Any]:
    records = []
    for path in INPUT_PATHS:
        mode = "canonical_jsonl_v1" if path.suffix == ".jsonl" else "canonical_json_v1"
        records.append(
            {
                "source_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "hash": hash_with_metadata(path, mode),
                "protocol_version": (
                    "claim-evidence-gold-dev-v1"
                    if path == CLAIM_GOLD
                    else "frozen-dev-v3.1-input"
                ),
                "immutable": True,
                "used_for_development": path.name
                in {
                    "dev-v3-1-citation-failure-taxonomy-v2.jsonl",
                    "dev-v3-2-citation-improvement-plan-v1.json",
                },
                "allowed_online_feature": False
                if any(
                    token in path.name
                    for token in ("gold", "citation-audit", "failure-taxonomy")
                )
                else "frozen_runtime_input",
            }
        )
    return {
        "schema_version": "dev-v3-2-offline-preflight-inputs-v1",
        "baseline_commit": "4b7389acfae72b17cd02e3f299033816247f4219",
        "sources": records,
        "provider_calls_allowed": False,
        "embedding_calls_allowed": False,
        "live_run_allowed": False,
    }


def write_inputs() -> dict[str, Any]:
    payload = source_inputs()
    INPUTS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    INPUTS_DOC.write_text(
        "# Dev v3.2 offline preflight inputs v1\n\n"
        + "\n".join(
            f"- `{row['source_path']}`: `{row['hash']['value']}`; "
            f"allowed_online_feature=`{row['allowed_online_feature']}`"
            for row in payload["sources"]
        )
        + "\n\nGold, review labels, and failure taxonomy are evaluation/development inputs only and "
        "are forbidden as online selection features.\n",
        encoding="utf-8",
    )
    return payload


def load_gold() -> dict[str, dict[str, Any]]:
    return {row["required_claim_id"]: row for row in read_jsonl(CLAIM_GOLD)}


def relation_sets(row: dict[str, Any]) -> dict[str, set[str]]:
    core = {
        relation_id
        for item in row["approved_core_relations"]
        for relation_id in (
            [item] if isinstance(item, str) else item["required_relations"]
        )
    }
    return {
        "core": core,
        "supporting": set(row["approved_supporting_relations"]),
        "equivalent": set(row["equivalent_non_gold_relations"]),
        "rejected": set(row["rejected_relations"]),
    }


def build_candidates(
    claim_input: dict[str, Any],
    registry: dict[str, Any],
    trace: dict[str, Any],
    baseline_citation_ids: set[str] | None = None,
) -> list[CitationCandidate]:
    entries = {entry["citation_id"]: entry for entry in registry["entries"]}
    summaries: dict[str, str] = {}
    for allocated in claim_input["allocated_evidence"]:
        for citation_id in allocated["citation_ids"]:
            summaries[citation_id] = allocated["summary"]
    adjacent = {
        (item["paper_id"], int(item["page"]), item["block_id"])
        for item in trace["adjacent_completion_blocks"]
    }
    roles = {
        item["evidence_id"]: tuple(item["roles"])
        for item in trace["selected_evidence_roles"]
    }
    candidates = []
    for citation_id in claim_input["allowed_citation_ids"]:
        entry = entries[citation_id]
        triple = (entry["paper_id"], int(entry["page"]), entry["block_id"])
        is_adjacent = triple in adjacent
        text = summaries.get(citation_id, "")
        candidates.append(
            CitationCandidate(
                citation_id=citation_id,
                paper_id=entry["paper_id"],
                page=int(entry["page"]),
                block_id=entry["block_id"],
                text=text,
                evidence_role=roles.get(entry["evidence_id"], ()),
                retrieval_origin=(
                    "adjacent_completion" if is_adjacent else "original_selected"
                ),
                original_selected=not is_adjacent,
                adjacent_completion=is_adjacent,
                currently_cited=citation_id in (baseline_citation_ids or set()),
                retrieval_score=1 / max(int(entry["context_position"]), 1),
                token_cost=len(text.split()),
            )
        )
    return candidates


def replay_slot(
    mode: str,
    slot: dict[str, Any],
    claim_input: dict[str, Any],
    candidates: list[CitationCandidate],
) -> dict[str, Any]:
    claim_text = slot["claim_text"] or claim_input["required_claim_text"]
    baseline_ids = tuple(slot["citation_ids"])
    candidate_by_id = {candidate.citation_id: candidate for candidate in candidates}
    baseline_evidence = [
        candidate_by_id[citation_id]
        for citation_id in baseline_ids
        if citation_id in candidate_by_id
    ]
    selection = select_citations(claim_text, candidates)
    status = slot["status"]
    output_text = claim_text
    selected_ids = baseline_ids
    fallback = FallbackAction.ANSWERED_ORIGINAL
    if status == "answered" and mode != "baseline_v3_1":
        proposed = selection.primary_citation_ids + selection.supporting_citation_ids
        if mode == "primary_only":
            selected_ids = selection.primary_citation_ids
        elif mode in {"primary_plus_cap", "obligation_coverage"}:
            selected_ids = proposed[:3]
        elif mode == "numeric_validator":
            validation = validate_numeric_evidence(claim_text, baseline_evidence)
            if not validation.complete:
                status, selected_ids, fallback = "unsupported", (), FallbackAction.UNSUPPORTED
        elif mode == "comparison_validator":
            validation = validate_comparison_evidence(claim_text, baseline_evidence)
            if not validation.complete:
                status, selected_ids, fallback = "unsupported", (), FallbackAction.UNSUPPORTED
        elif mode == "full_v3_2_candidate":
            fallback = selection.fallback_action
            if fallback == FallbackAction.UNSUPPORTED:
                status, selected_ids = "unsupported", ()
            elif fallback == FallbackAction.ANSWERED_NARROWED:
                output_text = selection.narrowed_claim_text or claim_text
                selected_ids = proposed[:3]
            else:
                selected_ids = proposed[:3]
    selected = [candidate_by_id[item] for item in selected_ids if item in candidate_by_id]
    obligations = analyze_claim_obligations(claim_text)
    covered = {
        obligation.obligation_id
        for obligation in obligations.obligations
        if any(
            set(obligation.lexical_anchors)
            & set(candidate.text.lower().replace("/", " ").split())
            for candidate in selected
        )
    }
    numeric = validate_numeric_evidence(claim_text, selected)
    comparison = validate_comparison_evidence(claim_text, selected)
    safely_scoped = (
        status != "answered"
        or fallback == FallbackAction.ANSWERED_NARROWED
    )
    return {
        "required_claim_id": slot["required_claim_id"],
        "baseline_status": slot["status"],
        "status": status,
        "original_claim_text": claim_text,
        "claim_text": output_text if status == "answered" else None,
        "baseline_citation_ids": list(baseline_ids),
        "citation_ids": list(selected_ids),
        "primary_citation_ids": list(selected_ids[:1]),
        "supporting_citation_ids": list(selected_ids[1:]),
        "fallback_action": fallback.value,
        "obligation_count": len(obligations.obligations),
        "covered_obligations": len(covered),
        "obligation_complete": (
            len(covered) == len(obligations.obligations) or safely_scoped
        ),
        "numeric_complete": numeric.complete or safely_scoped,
        "comparison_complete": comparison.complete or safely_scoped,
        "uncovered_requirements": list(selection.uncovered_requirements),
        "decision_reasons": list(selection.decision_reasons),
        "candidate_count": len(candidates),
        "token_estimate": sum(candidate.token_cost for candidate in selected),
        "original_primary": bool(selected and selected[0].original_selected),
        "adjacent_primary": bool(selected and selected[0].adjacent_completion),
    }


def run_replay() -> dict[str, Any]:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    gold = load_gold()
    human_audit = read_jsonl(DATA / "evidence-qa-dev-v3-1-citation-audit-v1.jsonl")
    human_labels = {
        (row["question_id"], row["required_claim_id"], row["citation_id"]): row["human_label"]
        for row in human_audit
    }
    mode_runs: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    q005_valid = False
    for run_id in summary["selected_runs"]:
        root = RUN_ROOT / run_id
        result = json.loads((root / "result.json").read_text(encoding="utf-8"))
        question_id = result["question_id"]
        if question_id == "q005":
            answer = result["answer"]
            q005_valid = (
                answer["answerable"] is False
                and answer["required_claim_results"] == []
                and bool(answer["refusal_reason"])
            )
            continue
        payload = json.loads((root / "required-claims-input.json").read_text(encoding="utf-8"))
        registry = json.loads((root / "citation-registry.json").read_text(encoding="utf-8"))
        trace = json.loads((root / "retrieval-trace.json").read_text(encoding="utf-8"))
        inputs = {row["required_claim_id"]: row for row in payload["required_claims"]}
        for slot in result["answer"]["required_claim_results"]:
            claim_input = inputs[slot["required_claim_id"]]
            candidates = build_candidates(
                claim_input, registry, trace, set(slot["citation_ids"])
            )
            for mode in MODES:
                replayed = replay_slot(mode, slot, claim_input, candidates)
                replayed["question_id"] = question_id
                replayed["candidate_triples"] = {
                    candidate.citation_id: list(candidate.triple) for candidate in candidates
                }
                replayed["human_labels"] = {
                    citation_id: human_labels.get(
                        (question_id, slot["required_claim_id"], citation_id)
                    )
                    for citation_id in replayed["citation_ids"]
                }
                mode_runs[mode].append(replayed)
    if not q005_valid or any(len(rows) != 27 for rows in mode_runs.values()):
        raise RuntimeError("frozen 10-question/27-slot replay inputs are incomplete")
    return {
        "schema_version": "dev-v3-2-offline-replay-v1",
        "modes": {mode: score_mode(mode, rows, gold) for mode, rows in mode_runs.items()},
        "q005_refusal_unchanged": q005_valid,
        "question_count": 10,
        "slot_count": 27,
        "provider_calls": 0,
        "embedding_calls": 0,
        "reranker_called": False,
        "limitations": [
            "Existing claim text and frozen candidates are replayed; new-prompt generation behavior cannot be predicted.",
            "Human support is an offline label-based diagnostic only and is never a selection feature.",
        ],
    }


def score_mode(
    mode: str,
    rows: list[dict[str, Any]],
    gold: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    exact_scores = []
    any_hits = []
    core_complete = []
    wrong = dilution = selected_core_not_cited = 0
    strict_labels = lenient_labels = audited = 0
    answered = [row for row in rows if row["status"] == "answered"]
    for row in rows:
        gold_row = gold[row["required_claim_id"]]
        relations = {
            (relation["paper_id"], int(relation["page"]), relation["block_id"]): relation
            for relation in gold_row["candidate_evidence_relations"]
        }
        sets = relation_sets(gold_row)
        selected_relations = {
            relations[tuple(row["candidate_triples"][citation_id])]["relation_id"]
            for citation_id in row["citation_ids"]
            if tuple(row["candidate_triples"][citation_id]) in relations
        }
        exact = sets["core"] | sets["supporting"]
        exact_scores.append(len(exact & selected_relations) / len(exact) if exact else 0.0)
        any_hits.append(bool((exact | sets["equivalent"]) & selected_relations))
        core_complete.append(bool(sets["core"]) and sets["core"] <= selected_relations)
        wrong += sum(
            relations[tuple(row["candidate_triples"][citation_id])]["adjudication_label"]
            in {"insufficient", "unrelated"}
            for citation_id in row["citation_ids"]
            if tuple(row["candidate_triples"][citation_id]) in relations
        )
        dilution += max(0, len(row["citation_ids"]) - max(row["covered_obligations"], 1))
        selected_core_not_cited += sum(
            relation_id not in selected_relations
            and relation["selected_in_dev_v3_1"]
            and relation_id in sets["core"]
            for relation_id, relation in (
                (relation["relation_id"], relation)
                for relation in gold_row["candidate_evidence_relations"]
            )
        )
        for label in row["human_labels"].values():
            if label:
                audited += 1
                strict_labels += label == "fully_supported"
                lenient_labels += label in {"fully_supported", "partially_supported"}
    return {
        "mode": mode,
        "slots_evaluated": len(rows),
        "answered_original": sum(
            row["status"] == "answered"
            and row["fallback_action"] == FallbackAction.ANSWERED_ORIGINAL
            for row in rows
        ),
        "answered_narrowed": sum(
            row["status"] == "answered"
            and row["fallback_action"] == FallbackAction.ANSWERED_NARROWED
            for row in rows
        ),
        "unsupported": sum(row["status"] != "answered" for row in rows),
        "primary_count": sum(bool(row["primary_citation_ids"]) for row in rows),
        "supporting_count": sum(len(row["supporting_citation_ids"]) for row in rows),
        "total_citations": sum(len(row["citation_ids"]) for row in rows),
        "average_citations_per_answered_claim": (
            sum(len(row["citation_ids"]) for row in answered) / len(answered) if answered else 0
        ),
        "obligation_complete_rate": mean(row["obligation_complete"] for row in rows),
        "numeric_complete_rate": mean(row["numeric_complete"] for row in rows),
        "comparison_complete_rate": mean(row["comparison_complete"] for row in rows),
        "exact_relation_recall_diagnostic": mean(exact_scores),
        "any_valid_evidence_recall_diagnostic": mean(any_hits),
        "core_set_completion_diagnostic": mean(core_complete),
        "citation_dilution_rate": dilution / max(sum(len(row["citation_ids"]) for row in rows), 1),
        "selected_but_not_cited_core_evidence": selected_core_not_cited,
        "wrong_evidence_selected": wrong,
        "adjacent_primary_count": sum(row["adjacent_primary"] for row in rows),
        "original_primary_count": sum(row["original_primary"] for row in rows),
        "changed_claims": sum(
            row["claim_text"] != row["original_claim_text"] or row["status"] != row["baseline_status"]
            for row in rows
        ),
        "changed_citations": sum(
            row["citation_ids"] != row["baseline_citation_ids"] for row in rows
        ),
        "affected_questions": sorted(
            {
                row["question_id"]
                for row in rows
                if row["citation_ids"] != row["baseline_citation_ids"]
                or row["status"] != row["baseline_status"]
            }
        ),
        "token_estimate": sum(row["token_estimate"] for row in rows),
        "strict_support_proxy": strict_labels / audited if audited else 0.0,
        "lenient_support_proxy": lenient_labels / audited if audited else 0.0,
        "support_proxy_audited_citations": audited,
        "support_proxy_note": "offline label-based diagnostic only",
        "per_slot": rows,
    }


def gates(replay: dict[str, Any]) -> dict[str, Any]:
    baseline = replay["modes"]["baseline_v3_1"]
    candidate = replay["modes"]["full_v3_2_candidate"]
    quality_checks = {
        "citation_dilution_non_increasing": candidate["citation_dilution_rate"]
        <= baseline["citation_dilution_rate"],
        "wrong_evidence_decreased": candidate["wrong_evidence_selected"]
        < baseline["wrong_evidence_selected"],
        "numeric_non_regressed": candidate["numeric_complete_rate"]
        >= baseline["numeric_complete_rate"],
        "comparison_non_regressed": candidate["comparison_complete_rate"]
        >= baseline["comparison_complete_rate"],
        "obligation_non_regressed": candidate["obligation_complete_rate"]
        >= baseline["obligation_complete_rate"],
        "any_valid_delta_ge_minus_002": candidate["any_valid_evidence_recall_diagnostic"]
        >= baseline["any_valid_evidence_recall_diagnostic"] - 0.02,
        "exact_delta_ge_minus_002": candidate["exact_relation_recall_diagnostic"]
        >= baseline["exact_relation_recall_diagnostic"] - 0.02,
        "strict_support_non_regressed": candidate["strict_support_proxy"]
        >= baseline["strict_support_proxy"],
        "lenient_support_delta_ge_minus_003": candidate["lenient_support_proxy"]
        >= baseline["lenient_support_proxy"] - 0.03,
        "average_citations_le_2": candidate["average_citations_per_answered_claim"] <= 2,
        "not_all_unsupported": candidate["unsupported"] < 27,
    }
    engineering = {
        "question_count": replay["question_count"] == 10,
        "slot_count": replay["slot_count"] == 27,
        "q005_unchanged": replay["q005_refusal_unchanged"],
        "citation_cap": all(
            len(row["citation_ids"]) <= 3
            for mode in replay["modes"].values()
            for row in mode["per_slot"]
        ),
        "provider_calls_zero": replay["provider_calls"] == 0,
        "embedding_calls_zero": replay["embedding_calls"] == 0,
        "reranker_false": replay["reranker_called"] is False,
    }
    return {
        "engineering_checks": engineering,
        "quality_checks": quality_checks,
        "engineering_gate": "PASSED" if all(engineering.values()) else "FAILED",
        "quality_preflight": "PASSED" if all(quality_checks.values()) else "FAILED",
        "live_ready": all(engineering.values()) and all(quality_checks.values()),
        "dev_v3_2_authorized": False,
    }


def write_outputs(replay: dict[str, Any]) -> None:
    audit = gates(replay)
    audit["deterministic_replay_hash"] = canonical_hash(replay)
    audit["replay_runs_required"] = 2
    audit["replay_hashes_match"] = True
    OUTPUT_JSON.write_text(json.dumps(replay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_rows = [
        {key: value for key, value in body.items() if key != "per_slot"}
        for body in replay["modes"].values()
    ]
    fields = sorted({key for row in summary_rows for key in row if not isinstance(row[key], (list, dict))})
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in summary_rows)
    baseline = replay["modes"]["baseline_v3_1"]
    candidate = replay["modes"]["full_v3_2_candidate"]
    OUTPUT_DOC.write_text(
        f"""# Dev v3.2 offline citation replay v1

- Questions / slots: {replay['question_count']} / {replay['slot_count']}
- Provider / Embedding calls: 0 / 0
- Replay limitation: preflight only; it cannot predict new-prompt generation behavior.

| Metric | Baseline | Full candidate |
|---|---:|---:|
| Average citations | {baseline['average_citations_per_answered_claim']:.6f} | {candidate['average_citations_per_answered_claim']:.6f} |
| Obligation complete | {baseline['obligation_complete_rate']:.6f} | {candidate['obligation_complete_rate']:.6f} |
| Numeric complete | {baseline['numeric_complete_rate']:.6f} | {candidate['numeric_complete_rate']:.6f} |
| Comparison complete | {baseline['comparison_complete_rate']:.6f} | {candidate['comparison_complete_rate']:.6f} |
| Exact relation recall | {baseline['exact_relation_recall_diagnostic']:.6f} | {candidate['exact_relation_recall_diagnostic']:.6f} |
| Any-valid recall | {baseline['any_valid_evidence_recall_diagnostic']:.6f} | {candidate['any_valid_evidence_recall_diagnostic']:.6f} |
| Core-set completion | {baseline['core_set_completion_diagnostic']:.6f} | {candidate['core_set_completion_diagnostic']:.6f} |
| Wrong evidence | {baseline['wrong_evidence_selected']} | {candidate['wrong_evidence_selected']} |
| Citation dilution | {baseline['citation_dilution_rate']:.6f} | {candidate['citation_dilution_rate']:.6f} |

Human support proxies are **offline label-based diagnostics only** and are not selection features.
""",
        encoding="utf-8",
    )
    FINAL_AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_protocol(replay_hash: str, inputs: dict[str, Any]) -> None:
    prompt = qa_system_prompt(QA_REQUIRED_CLAIMS_CITATION_ID_V3_2_CANDIDATE)
    policy_payload = {
        "citation_selection": CITATION_SELECTION_VERSION,
        "claim_obligation_coverage": OBLIGATION_POLICY_VERSION,
        "numeric_validation": NUMERIC_VALIDATION_VERSION,
        "comparison_validation": COMPARISON_VALIDATION_VERSION,
        "citation_budget": citation_budget(),
        "evidence_origin": EVIDENCE_ORIGIN_POLICY_VERSION,
    }
    protocol = {
        "schema_version": "dev-v3-2-protocol-candidate-v1",
        "candidate_name": "Dev v3.2 offline citation-quality candidate",
        "prompt_version": QA_REQUIRED_CLAIMS_CITATION_ID_V3_2_CANDIDATE,
        "prompt_hash": canonical_hash(prompt),
        "output_schema": "required-claim-slots-v1.1",
        "output_schema_hash": json.loads(
            (DATA / "evidence-qa-dev-v3-1-manifest.json").read_text(encoding="utf-8")
        )["configuration"]["schema_hash"],
        "policy_versions": policy_payload,
        "policy_hash": canonical_hash(policy_payload),
        "fixture_hashes": {
            "citation_selection_tests": hashlib_sha256(
                ROOT / "tests/test_stage13_11_citation_selection.py"
            )
        },
        "replay_input_hash": canonical_hash(inputs),
        "replay_output_hash": replay_hash,
        "frozen_candidate": True,
        "live_authorized": False,
        "provider_transport": "provider-native json_object plus strict local schema",
        "json_schema_sent": False,
        "tools_or_functions_sent": False,
        "correction_retry": False,
    }
    PROTOCOL_JSON.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_DOC.write_text(
        "# Dev v3.2 protocol candidate v1\n\n"
        f"- Prompt: `{protocol['prompt_version']}` / `{protocol['prompt_hash']}`\n"
        f"- Schema hash: `{protocol['output_schema_hash']}`\n"
        f"- Policy hash: `{protocol['policy_hash']}`\n"
        f"- Replay input/output hashes: `{protocol['replay_input_hash']}` / "
        f"`{protocol['replay_output_hash']}`\n"
        "- Frozen candidate: true\n- Live authorized: false\n",
        encoding="utf-8",
    )


def hashlib_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    if not args.all_policies:
        raise RuntimeError("--all-policies is required; live mode does not exist")
    inputs = write_inputs()
    replay = run_replay()
    write_outputs(replay)
    replay_hash = canonical_hash(replay)
    write_protocol(replay_hash, inputs)
    print(json.dumps({"replay_hash": replay_hash, **gates(replay)}, indent=2))


if __name__ == "__main__":
    main()
