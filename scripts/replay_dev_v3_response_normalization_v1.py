# ruff: noqa: E501,E701,E702,I001
"""Offline-only deterministic replay of the ten Stage 13.5 raw responses."""

from __future__ import annotations

import csv
import json

from paper_research.generation.citation_registry import CitationRegistry
from paper_research.generation.required_claim_output import RequiredClaimValidationError, parse_and_validate_required_claim_response
from paper_research.generation.response_normalization import NORMALIZATION_SCHEMA_VERSION, normalize_response

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DOCS
    from scripts.evidence_qa_dev_v3_lib import RUN_ROOT
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DATA, DOCS  # type: ignore[no-redef]
    from evidence_qa_dev_v3_lib import RUN_ROOT  # type: ignore[no-redef]

OUTPUT=DATA/"dev-v3-response-normalization-replay-v1.json"
OUTPUT_CSV=DATA/"dev-v3-response-normalization-replay-v1.csv"
OUTPUT_DOC=DOCS/"dev-v3-response-normalization-replay-v1.md"


def replay(run_dir):
    result=json.loads((run_dir/"result.json").read_text(encoding="utf-8")); payload=json.loads((run_dir/"required-claims-input.json").read_text(encoding="utf-8")); registry=CitationRegistry.model_validate_json((run_dir/"citation-registry.json").read_text(encoding="utf-8")); provider=json.loads((run_dir/"raw-provider-response.json").read_text(encoding="utf-8")); content=provider["choices"][0]["message"]["content"]
    expected=[row["required_claim_id"] for row in payload["required_claims"]]; allowed={row["required_claim_id"]:set(row["allowed_citation_ids"]) for row in payload["required_claims"]}
    try: raw=json.loads(content); raw_valid=True
    except json.JSONDecodeError: raw=None; raw_valid=False
    normalization=normalize_response(raw,question_id=result["question_id"],expected_claim_ids=expected) if raw_valid else None
    normalized_pass=False; validation_error=None; slots=[]
    if normalization and normalization.accepted and normalization.payload is not None:
        try:
            output=parse_and_validate_required_claim_response(json.dumps(normalization.payload),expected_claim_ids=expected,registry=registry,allowed_by_claim=allowed,expected_registry_hash=registry.registry_hash); normalized_pass=True; slots=output.required_claim_results
        except RequiredClaimValidationError as exc: validation_error=str(exc)
    answered=sum(slot.status.value=="answered" for slot in slots); unsupported=sum(slot.status.value=="unsupported" for slot in slots); na=sum(slot.status.value=="not_applicable" for slot in slots)
    return {"question_id":result["question_id"],"run_id":result["run_id"],"official_status":result["status"],"raw_schema_pass":False,"normalization_attempted":bool(normalization),"normalization_accepted":bool(normalization and normalization.accepted),"normalization_status":normalization.status if normalization else "malformed_json","operations_applied":list(normalization.operations) if normalization else [],"normalized_payload_hash":normalization.normalized_payload_hash if normalization else None,"normalization_schema_version":NORMALIZATION_SCHEMA_VERSION,"normalization_risk_level":normalization.risk_level if normalization else "none","normalized_schema_pass":normalized_pass,"required_claim_slot_completeness":len(slots)==len(expected) if normalized_pass else False,"citation_registry_validation":normalized_pass,"claim_local_citation_validation":normalized_pass,"answerability_validation":normalized_pass,"semantic_information_loss":normalization.semantic_information_loss if normalization else False,"answered_slots":answered,"unsupported_slots":unsupported,"not_applicable_slots":na,"diagnostic_covered_claims":answered,"required_claim_denominator":len(expected),"validation_error":validation_error or (normalization.reason if normalization else "malformed JSON is not repaired")}


def main():
    rows=sorted((replay(path.parent) for path in RUN_ROOT.glob("live-dev-v3-*/result.json")),key=lambda row:row["question_id"])
    if len(rows)!=10: raise RuntimeError("expected ten frozen runs")
    covered=sum(row["diagnostic_covered_claims"] for row in rows); denom=sum(row["required_claim_denominator"] for row in rows); accepted=sum(row["normalized_schema_pass"] for row in rows)
    metrics={"strict_raw":{"schema_success":0.0,"required_claim_coverage":{"numerator":0,"denominator":27,"rate":0.0}},"deterministically_normalized_diagnostic":{"schema_success":accepted/10,"slot_completeness":sum(row["required_claim_slot_completeness"] for row in rows)/10,"required_claim_coverage":{"numerator":covered,"denominator":denom,"rate":covered/denom if denom else 0},"answered_slots":sum(row["answered_slots"] for row in rows),"unsupported_slots":sum(row["unsupported_slots"] for row in rows),"not_applicable_slots":sum(row["not_applicable_slots"] for row in rows),"citation_precision":0.0,"citation_recall":0.0,"unknown_citation_id":0,"cross_claim_citation":0,"normalization_rejection_count":sum(not row["normalization_accepted"] for row in rows)}}
    payload={"schema_version":"dev-v3-response-normalization-replay-v1","evaluation_mode":"diagnostic_replay","official_stage13_5_modified":False,"formal_gate_eligible":False,"metrics":metrics,"rows":rows}
    OUTPUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    with OUTPUT_CSV.open("w",encoding="utf-8",newline="") as stream:
        flat=[{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows]; writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    table="\n".join(f"| {row['question_id']} | {row['operations_applied']} | {row['normalization_status']} | {row['normalized_schema_pass']} |" for row in rows)
    OUTPUT_DOC.write_text(f"# Dev v3 Response Normalization Replay\n\n- Strict raw schema success: 0/10\n- Deterministically normalized schema success: {accepted}/10\n- Strict/diagnostic coverage: 0/27 / {covered}/{denom}\n- Rejections: {metrics['deterministically_normalized_diagnostic']['normalization_rejection_count']}\n\n| Question | Operations considered | Status | Schema pass |\n|---|---|---|---|\n{table}\n\nThis is diagnostic replay only. It neither replaces Stage 13.5 nor permits Full QA. No semantic repair, fuzzy matching, missing-slot creation, free-triple conversion, or LLM call occurred.\n",encoding="utf-8")
    print(json.dumps({"raw_schema_success":0,"normalized_schema_success":accepted,"diagnostic_coverage":f"{covered}/{denom}"}))


if __name__=="__main__": main()
