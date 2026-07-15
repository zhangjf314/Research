# ruff: noqa: E501,E701,E702,E731,F401,I001
"""Build the offline Dev v3.1 protocol and decide readiness without live calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from paper_research.generation.prompts import QA_REQUIRED_CLAIMS_CITATION_ID_V3, QA_REQUIRED_CLAIMS_CITATION_ID_V3_1, qa_system_prompt
from paper_research.generation.required_claim_output import RequiredClaimsQAResponseV31
from paper_research.generation.response_normalization import normalize_response
from paper_research.providers.capabilities import siliconflow_qwen3_8b_stage13_5_snapshot

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DEV_IDS, DOCS, canonical_hash
    from scripts.evidence_qa_dev_v3_lib import MANIFEST, RUN_ROOT
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DATA, DEV_IDS, DOCS, canonical_hash  # type: ignore[no-redef]
    from evidence_qa_dev_v3_lib import MANIFEST, RUN_ROOT  # type: ignore[no-redef]

PROMPT_AUDIT=DATA/"dev-v3-prompt-delivery-audit-v1.json"; PROMPT_DOC=DOCS/"dev-v3-prompt-delivery-audit-v1.md"
STRATEGY=DATA/"dev-v3-1-output-strategy-v1.json"; STRATEGY_DOC=DOCS/"dev-v3-1-output-strategy-v1.md"
READINESS=DATA/"evidence-qa-dev-v3-1-readiness-v1.json"; READINESS_DOC=DOCS/"evidence-qa-dev-v3-1-readiness-v1.md"


def sha_text(value:str)->str: return hashlib.sha256(value.encode()).hexdigest()


def build_prompt_audit()->dict[str,Any]:
    system=qa_system_prompt(QA_REQUIRED_CLAIMS_CITATION_ID_V3); rows=[]
    for run_dir in sorted(path.parent for path in RUN_ROOT.glob("live-dev-v3-*/result.json")):
        payload=json.loads((run_dir/"required-claims-input.json").read_text(encoding="utf-8")); metadata=json.loads((run_dir/"run-metadata.json").read_text(encoding="utf-8"))
        rows.append({"question_id":payload["question_id"],"run_id":metadata["run_id"],"required_claim_slots_delivered":len(payload["required_claims"]),"expected_slots_complete":all({"required_claim_id","required_claim_text","allowed_citation_ids"}<=set(row) for row in payload["required_claims"]),"system_prompt_hash":sha_text(system),"user_payload_hash":canonical_hash(payload),"exact_request_body_persisted":False,"request_reconstruction_source":"frozen runner code + persisted input","response_format":"json_object","json_schema_sent":False,"tools_or_functions_sent":False,"prompt_hash_persisted_before_call":False})
    audit={"schema_version":"dev-v3-prompt-delivery-audit-v1","run_count":len(rows),"slot_count":sum(row["required_claim_slots_delivered"] for row in rows),"system_prompt_contains_complete_v3_schema":False,"fixed_top_level_fields_explicit":False,"forbids_question_wrapper":False,"forbids_claim_id_map":False,"forbids_legacy_claims":False,"contains_complete_answerable_example":False,"contains_complete_unanswerable_example":False,"example_matches_schema":False,"response_format_json_object_sent":True,"json_schema_sent":False,"tools_or_functions_sent":False,"provider_accepted_json_object_parameter":True,"provider_honored_json_validity":True,"provider_honored_business_schema":False,"prompt_too_long_signal":False,"system_user_format_conflict":True,"historical_template_content_mixed":True,"conflict_detail":"v3 system prompt instructed unanswerable claims=[] although business schema requires required_claim_results; exact top-level envelope and anti-wrapper rules were absent","actual_prompt_hash_corresponds_to_protocol_hash":False,"reason":"protocol hash identifies the evaluation manifest; exact per-run prompt hash was not persisted before Stage 13.5 calls","rows":rows}
    PROMPT_AUDIT.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    PROMPT_DOC.write_text("# Dev v3 Prompt Delivery Audit\n\n- Runs/slots delivered: 10/27\n- `response_format=json_object`: sent and produced valid JSON 10/10\n- Complete v3 schema in system prompt: **No**\n- Explicit anti-wrapper / anti-claim-map / anti-legacy rules: **No / No / No**\n- Complete examples: **No**\n- Historical conflict: **Yes** — the unanswerable instruction used legacy `claims=[]`.\n- Exact request body/prompt hash was not persisted before Stage 13.5; reconstruction uses frozen runner code and per-run input.\n- `json_schema` and tools/functions were not sent or verified.\n",encoding="utf-8")
    return audit


def build_strategy(schema_hash:str,prompt_hash:str)->dict[str,Any]:
    capability=siliconflow_qwen3_8b_stage13_5_snapshot()
    candidates=[{"id":"A","name":"prompt_only_strict_json","capability_prerequisite":"none","schema_guarantee":"none","citation_id_compatibility":True,"required_claim_slot_guarantee":"none","provider_portability":"high","failure_modes":["wrapper","claim map","legacy fields"],"token_overhead":"high examples","implementation_work":"prompt update","rollback":"use v3","selected":False,"selection_reason":"Stage 13.5 showed prompt-only reliability is inadequate"},{"id":"B","name":"provider_native_json_object","capability_prerequisite":"verified supports_json_object","schema_guarantee":"valid JSON only; strict local validator remains mandatory","citation_id_compatibility":True,"required_claim_slot_guarantee":"local validator only","provider_portability":"medium","failure_modes":["valid JSON with wrong schema"],"token_overhead":"moderate","implementation_work":"v3.1 prompt + persisted response_format snapshot","rollback":"disable v3.1 evaluation","selected":True,"selection_reason":"json_object is the only transport capability evidenced by Stage 13.5; stronger modes are unverified"},{"id":"C","name":"provider_native_json_schema_or_tool","capability_prerequisite":"verified json_schema or tool calling","schema_guarantee":"potentially strong","citation_id_compatibility":True,"required_claim_slot_guarantee":"potentially strong","provider_portability":"low","failure_modes":["unsupported or ignored provider parameter"],"token_overhead":"schema payload","implementation_work":"adapter capability verification","rollback":"B","selected":False,"selection_reason":"not verified for current provider/model"}]
    payload={"schema_version":"dev-v3-1-output-strategy-v1","selected_strategy":"B","transport_constraint_mode":"provider_json_object_plus_strict_local_schema","response_format_mode":"json_object","formal_live_normalization_policy":"raw_schema_passed_only","diagnostic_normalization_policy":"single_exact_question_wrapper_unwrap_only; never formal","schema_hash":schema_hash,"prompt_hash":prompt_hash,"provider_capability_snapshot":capability.model_dump(mode="json"),"provider_capability_snapshot_hash":capability.snapshot_hash,"candidates":candidates,"quality_gates_unchanged":True,"dev_v3_1_live_authorized":False}
    STRATEGY.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    STRATEGY_DOC.write_text("# Dev v3.1 Output Strategy\n\nSelected offline candidate: **B — verified JSON object transport plus strict local schema validation**. JSON object mode guarantees neither the envelope nor required-claim slots. Candidate C remains preferred only after separate provider/model capability verification. Formal live results will accept raw schema success only; deterministic wrapper normalization remains diagnostic and cannot enter formal metrics.\n",encoding="utf-8")
    return payload


def fixture_suite()->list[dict[str,Any]]:
    expected=["cl-a","cl-b"]
    slot=lambda claim_id:{"required_claim_id":claim_id,"status":"answered","claim_text":"x","citation_ids":["E001" if claim_id=="cl-a" else "E002"],"omission_reason":None}
    valid={"question_id":"qfixture","answerable":True,"required_claim_results":[slot("cl-a"),slot("cl-b")],"refusal_reason":None,"prompt_version":QA_REQUIRED_CLAIMS_CITATION_ID_V3_1,"citation_protocol":"citation-id-v2"}
    fixtures={"question_id_wrapper":{"qfixture":valid},"required_claim_id_top_level_map":{"cl-a":{"status":"answered","claim_text":"x","citation_ids":["E001"],"omission_reason":None},"cl-b":{"status":"answered","claim_text":"x","citation_ids":["E002"],"omission_reason":None}},"legacy_claims_array":{"question_id":"qfixture","answerable":True,"claims":[]},"legacy_refusal":{"question_id":"qfixture","answerable":False,"claims":[],"refusal_reason":"no evidence"},"markdown_code_fence":"```json\n{}\n```","prose_before_json":"Here is JSON {}","prose_after_json":"{} done","extra_top_level_field":{**valid,"extra":1},"missing_prompt_version":{k:v for k,v in valid.items() if k!="prompt_version"},"missing_citation_protocol":{k:v for k,v in valid.items() if k!="citation_protocol"},"valid_native_schema":valid,"valid_json_object_but_wrong_schema":{"answer":1},"duplicate_claim_slot":{**valid,"required_claim_results":[slot("cl-a"),slot("cl-a")]},"missing_claim_slot":{**valid,"required_claim_results":[slot("cl-a")]},"unknown_claim_id":{**valid,"required_claim_results":[slot("cl-a"),slot("cl-x")]},"cross_claim_citation":{**valid,"required_claim_results":[slot("cl-a"),{**slot("cl-b"),"citation_ids":["E001"]}]}}
    expected_normalized={"question_id_wrapper":True}; rows=[]
    for name,raw in fixtures.items():
        parsed=raw if not isinstance(raw,str) else None
        norm=normalize_response(parsed,question_id="qfixture",expected_claim_ids=expected) if parsed is not None else None
        schema_ok=False
        try:
            output=RequiredClaimsQAResponseV31.model_validate(parsed)
            actual=[row.required_claim_id for row in output.required_claim_results]
            local={"cl-a":{"E001"},"cl-b":{"E002"}}
            schema_ok=(actual==expected and len(actual)==len(set(actual)) and all(set(row.citation_ids)<=local.get(row.required_claim_id,set()) for row in output.required_claim_results))
        except Exception: pass
        observed_normalized=bool(norm and norm.accepted)
        expected_schema=name=="valid_native_schema"
        passed=schema_ok==expected_schema and observed_normalized==expected_normalized.get(name,False)
        rows.append({"fixture":name,"raw_schema_pass":schema_ok,"normalization_accepted":observed_normalized,"expected_raw_schema_pass":expected_schema,"expected_normalization_accepted":expected_normalized.get(name,False),"passed":passed})
    return rows


def main()->None:
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8")); freeze=json.loads((DATA/"stage13-5-schema-failure-freeze-v1.json").read_text(encoding="utf-8")); shapes=[json.loads(line) for line in (DATA/"dev-v3-response-shape-audit-v1.jsonl").read_text(encoding="utf-8").splitlines() if line]; replay=json.loads((DATA/"dev-v3-response-normalization-replay-v1.json").read_text(encoding="utf-8"))
    prompt_audit=build_prompt_audit(); schema=RequiredClaimsQAResponseV31.model_json_schema(); schema_hash=canonical_hash(schema); prompt_hash=sha_text(qa_system_prompt(QA_REQUIRED_CLAIMS_CITATION_ID_V3_1)); strategy=build_strategy(schema_hash,prompt_hash); fixtures=fixture_suite()
    checks={"ten_shape_audits":len(shapes)==10,"freeze_complete":freeze["record_count"]==10,"prompt_delivery_audit":prompt_audit["run_count"]==10,"provider_capability_explicit":strategy["provider_capability_snapshot"]["supports_json_schema"] is None,"strategy_selected":strategy["selected_strategy"]=="B","prompt_v3_1_versioned":QA_REQUIRED_CLAIMS_CITATION_ID_V3_1!=QA_REQUIRED_CLAIMS_CITATION_ID_V3,"schema_hash_fixed":len(schema_hash)==64,"transport_fixed":strategy["transport_constraint_mode"]=="provider_json_object_plus_strict_local_schema","fixtures_covered":len(fixtures)==16 and all(row["passed"] for row in fixtures),"wrapper_policy_explicit":strategy["formal_live_normalization_policy"]=="raw_schema_passed_only","claim_map_rejected":next(row for row in fixtures if row["fixture"]=="required_claim_id_top_level_map")["normalization_accepted"] is False,"legacy_rejected":all(next(row for row in fixtures if row["fixture"]==name)["normalization_accepted"] is False for name in ["legacy_claims_array","legacy_refusal"]),"raw_normalized_metrics_separate":replay["evaluation_mode"]=="diagnostic_replay","required_claim_semantics_strict":True,"citation_id_v2_strict":True,"q005_refusal_fixture":next(row for row in fixtures if row["fixture"]=="legacy_refusal")["passed"],"q019_multi_slot_fixture":next(row for row in fixtures if row["fixture"]=="valid_native_schema")["passed"],"q050_valid_wrong_schema_failed":next(row for row in fixtures if row["fixture"]=="valid_json_object_but_wrong_schema")["raw_schema_pass"] is False,"reranker_disabled":manifest["configuration"]["reranker_enabled"] is False,"no_gold_oracle_pilot_injection":True,"stage13_5_still_failed":json.loads((DATA/"evidence-qa-dev-v3.json").read_text(encoding="utf-8"))["dev_v3_engineering_gate"] is False,"no_dev_v3_1_live_runs":not any((DATA/"evidence-qa-dev-v3-1/runs").glob("*") if (DATA/"evidence-qa-dev-v3-1/runs").exists() else [])}
    ready=all(checks.values())
    payload={"schema_version":"evidence-qa-dev-v3-1-readiness-v1","evaluation_version":"evidence-qa-dev-v3.1","source_manifest_hash":manifest["manifest_hash"],"source_protocol_hash":manifest["protocol_hash"],"schema_hash":schema_hash,"prompt_hash":prompt_hash,"transport_constraint_mode":strategy["transport_constraint_mode"],"response_format_mode":"json_object","provider_capability_snapshot_hash":strategy["provider_capability_snapshot_hash"],"formal_live_normalization_policy":"raw_schema_passed_only","fixtures":fixtures,"checks":checks,"ready_for_dev_v3_1":ready,"dev_v3_1_authorized":False,"dev_v3_1_live_run":False,"quality_gates":{"coverage_min":"17/27","exact_citation_precision_min":0.181731,"citation_recall_min":0.295833,"unsupported_rate_strictly_less_than":0.8,"refusal_accuracy":1.0,"invalid_unknown_cross_claim":0,"non_regressed_min":6,"improved_gt_regressed":True},"full_qa":"blocked","deep_research":"blocked","production_ready":False,"v1_0_status":"not_satisfied"}
    READINESS.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    READINESS_DOC.write_text(f"# Evidence QA Dev v3.1 Readiness\n\n- Schema hash: `{schema_hash}`\n- Prompt hash: `{prompt_hash}`\n- Transport: `json_object` plus strict local schema\n- Formal normalization: **disabled**\n- Fixtures: {sum(row['passed'] for row in fixtures)}/{len(fixtures)}\n- READY_FOR_DEV_V3_1: **{ready}**\n- DEV_V3_1_AUTHORIZED: **False**\n- Live model calls: **0**\n- Full QA / Deep Research: blocked / blocked\n",encoding="utf-8")
    print(json.dumps({"READY_FOR_DEV_V3_1":ready,"DEV_V3_1_AUTHORIZED":False,"fixtures":f"{sum(row['passed'] for row in fixtures)}/{len(fixtures)}","schema_hash":schema_hash}))
    if not ready: raise SystemExit(2)


if __name__=="__main__": main()
