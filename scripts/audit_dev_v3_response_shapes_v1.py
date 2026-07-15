# ruff: noqa: E501,E701,E702
"""Classify the ten immutable Stage 13.5 response shapes."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from paper_research.generation.response_normalization import normalize_response, schema_family

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DOCS
    from scripts.evidence_qa_dev_v3_lib import RUN_ROOT
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DATA, DOCS  # type: ignore[no-redef]
    from evidence_qa_dev_v3_lib import RUN_ROOT  # type: ignore[no-redef]

OUTPUT = DATA / "dev-v3-response-shape-audit-v1.jsonl"
OUTPUT_CSV = DATA / "dev-v3-response-shape-audit-v1.csv"
OUTPUT_DOC = DOCS / "dev-v3-response-shape-audit-v1.md"


def walk(value: Any) -> list[Any]:
    values = [value]
    if isinstance(value, dict):
        for child in value.values(): values.extend(walk(child))
    elif isinstance(value, list):
        for child in value: values.extend(walk(child))
    return values


def audit_run(run_dir: Path) -> dict[str, Any]:
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    payload = json.loads((run_dir / "required-claims-input.json").read_text(encoding="utf-8"))
    registry = json.loads((run_dir / "citation-registry.json").read_text(encoding="utf-8"))
    provider = json.loads((run_dir / "raw-provider-response.json").read_text(encoding="utf-8"))
    content = provider["choices"][0]["message"]["content"]
    expected = [row["required_claim_id"] for row in payload["required_claims"]]
    try: raw = json.loads(content); valid = True
    except json.JSONDecodeError: raw, valid = None, False
    keys = list(raw) if isinstance(raw, dict) else []
    family = "malformed_json" if not valid else schema_family(raw, result["question_id"], set(expected))
    all_values = walk(raw)
    all_dicts = [item for item in all_values if isinstance(item, dict)]
    all_keys = [str(key) for item in all_dicts for key in item]
    observed = sorted(set(expected) & set(all_keys))
    citations = sorted(set(re.findall(r"\bE\d{3}\b", content)))
    known = {item["citation_id"] for item in registry["entries"]}
    allowed = {row["required_claim_id"]: set(row["allowed_citation_ids"]) for row in payload["required_claims"]}
    cross = []
    for item in all_dicts:
        claim_id = item.get("required_claim_id")
        cited = item.get("citation_ids", item.get("citations", []))
        if claim_id in allowed and isinstance(cited, list): cross.extend(c for c in cited if c in known and c not in allowed[claim_id])
    statuses = [item.get("status") for item in all_dicts]
    normalized = normalize_response(raw, question_id=result["question_id"], expected_claim_ids=expected) if valid else None
    wrapper_key = keys[0] if len(keys) == 1 and isinstance(raw.get(keys[0]), dict) else None
    nested = raw.get(wrapper_key) if wrapper_key else raw
    answerability = next((item["answerable"] for item in all_dicts if "answerable" in item), None)
    refusal = next((item.get("refusal_reason") for item in all_dicts if "refusal_reason" in item), None)
    prompt = next((item.get("prompt_version") for item in all_dicts if "prompt_version" in item), None)
    protocol = next((item.get("citation_protocol") for item in all_dicts if "citation_protocol" in item), None)
    return {"question_id": result["question_id"], "run_id": result["run_id"], "raw_json_valid": valid, "top_level_type": type(raw).__name__ if valid else "malformed", "top_level_keys": keys, "wrapper_type": family if family == "question_id_wrapper" else None, "wrapper_key": wrapper_key, "nested_payload_type": type(nested).__name__, "detected_schema_family": family, "required_claim_ids_expected": expected, "required_claim_ids_observed": observed, "missing_claim_ids": sorted(set(expected)-set(observed)), "duplicate_claim_ids": [], "extra_claim_ids": sorted({key for key in all_keys if key.startswith("cl-")} - set(expected)), "answered_slots_observed": statuses.count("answered"), "unsupported_slots_observed": statuses.count("unsupported"), "not_applicable_slots_observed": statuses.count("not_applicable"), "citation_ids_observed": citations, "unknown_citation_ids": sorted(set(citations)-known), "cross_claim_citation_candidates": sorted(set(cross)), "free_triple_fields_detected": sorted({key for key in all_keys if key in {"paper_id","page","block_id"}}), "answerability_detected": answerability, "refusal_reason_detected": refusal, "prompt_version_detected": prompt, "citation_protocol_detected": protocol, "deterministic_normalization_possible": bool(normalized and normalized.accepted), "normalization_reason": normalized.reason if normalized else "malformed JSON is never repaired", "semantic_information_loss": normalized.semantic_information_loss if normalized else False, "official_status": result["status"]}


def main() -> None:
    rows = sorted((audit_run(path.parent) for path in RUN_ROOT.glob("live-dev-v3-*/result.json")), key=lambda row: row["question_id"])
    if len(rows) != 10: raise RuntimeError("expected exactly ten live Dev v3 runs")
    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as stream:
        flat = [{key: json.dumps(value, ensure_ascii=False) if isinstance(value,(list,dict)) else value for key,value in row.items()} for row in rows]
        writer=csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    counts=Counter(row["detected_schema_family"] for row in rows)
    table="\n".join(f"| {row['question_id']} | {row['detected_schema_family']} | {row['top_level_keys']} | {row['deterministic_normalization_possible']} |" for row in rows)
    OUTPUT_DOC.write_text(f"# Dev v3 Response Shape Audit\n\n- Records: 10\n- Valid JSON: {sum(row['raw_json_valid'] for row in rows)}/10\n- Schema families: `{dict(counts)}`\n- Deterministic normalization possible: {sum(row['deterministic_normalization_possible'] for row in rows)}/10\n\n| Question | Family | Top-level keys | Normalizable |\n|---|---|---|---|\n{table}\n\nOfficial status remains `validation_failed` for every record.\n", encoding="utf-8")
    print(json.dumps({"records":10,"valid_json":sum(row['raw_json_valid'] for row in rows),"families":counts,"normalizable":sum(row['deterministic_normalization_possible'] for row in rows)}))


if __name__ == "__main__": main()
