# Full QA Production Audit v1

- Status: `COMPLETED_WITH_FAILURES`
- Historical status: `COMPLETED_WITH_FAILURES`
- Superseded: `false`
- Production Full QA gate: `COMPLETED_WITH_FAILURES`
- Completed/failed: `41` / `9`
- Model: `siliconflow` / `Qwen/Qwen3-8B`
- Reranker: `disabled`
- Deep Research executed: `false`
- Answerable accuracy: `1.0`
- Refusal accuracy: `1.0`
- Required claim coverage: `0.264957`
- Citation precision / recall: `0.153846` / `0.126496`
- Citation ID validity: `1.0`
- Gold block retrieved rate: `0.333333`
- Tokens input/output/total: `279252` / `11380` / `290632`
- Estimated cost USD: `None` configured=`False`
- Latency mean/p50/p95 ms: `{'mean': 15069.057, 'p50': 15654.505, 'p95': 20692.741}`

This is a 50-item human-reviewed internal development evaluation, not a blind holdout.

## Failed items

- `q014`: {"error":{"code":"HTTP_503","message":{"code":"CLAIM_QA_CITATION_VALIDATION_ERROR","message":"SiliconFlow QA failed after 1 request(s): citation_validation:page","stage":"CLAIM_CITATION_VALIDATE","api_request_count":1,"retry_reasons":["citation_validation:page"],"rate_limit_events":0,"response_audit_path":"/app/artifacts/private/qa-response-audits/q014-full-qa-rerun-q014-1784458604.json"},"request_id":"906cc7b1-d5aa-4e33-86c3-9fa0d44a974f"}}
- `q020`: {"error":{"code":"HTTP_503","message":{"code":"CLAIM_QA_JSON_PARSE_ERROR","message":"SiliconFlow QA failed after 1 request(s): malformed_json","stage":"LLM_JSON_PARSE","api_request_count":1,"retry_reasons":["malformed_json"],"rate_limit_events":0,"response_audit_path":"/app/artifacts/private/qa-response-audits/q020-full-qa-rerun-q020-1784458716.json"},"request_id":"451b2f7c-cb49-40ff-92f9-ecee561e5a4b"}}
- `q029`: {"error":{"code":"HTTP_503","message":{"code":"CLAIM_QA_CITATION_VALIDATION_ERROR","message":"SiliconFlow QA failed after 1 request(s): citation_validation:page","stage":"CLAIM_CITATION_VALIDATE","api_request_count":1,"retry_reasons":["citation_validation:page"],"rate_limit_events":0,"response_audit_path":"/app/artifacts/private/qa-response-audits/q029-full-qa-rerun-q029-1784458934.json"},"request_id":"212b88d4-ce4c-47eb-9151-654828658b5e"}}
- `q031`: {"error":{"code":"HTTP_503","message":{"code":"CLAIM_QA_JSON_PARSE_ERROR","message":"SiliconFlow QA failed after 1 request(s): malformed_json","stage":"LLM_JSON_PARSE","api_request_count":1,"retry_reasons":["malformed_json"],"rate_limit_events":0,"response_audit_path":"/app/artifacts/private/qa-response-audits/q031-full-qa-rerun-q031-1784458954.json"},"request_id":"27c9407c-1a28-4254-bf95-dc5dc0b6215a"}}
- `q032`: {"error":{"code":"HTTP_503","message":{"code":"CLAIM_QA_JSON_PARSE_ERROR","message":"SiliconFlow QA failed after 1 request(s): malformed_json","stage":"LLM_JSON_PARSE","api_request_count":1,"retry_reasons":["malformed_json"],"rate_limit_events":0,"response_audit_path":"/app/artifacts/private/qa-response-audits/q032-full-qa-rerun-q032-1784459040.json"},"request_id":"a05165c9-22fb-4f68-90d2-b97345bad4cb"}}
- `q035`: {"error":{"code":"HTTP_503","message":{"code":"CLAIM_QA_PROVIDER_TIMEOUT","message":"SiliconFlow QA failed after 1 request(s): ReadTimeout","stage":"LLM_PROVIDER_TIMEOUT","api_request_count":1,"retry_reasons":["ReadTimeout"],"rate_limit_events":0,"response_audit_path":"/app/artifacts/private/qa-response-audits/q035-full-qa-rerun-q035-1784459162-provider-failure.json"},"request_id":"aea0e575-008b-432e-bbde-5e3bad9b6b43"}}
- `q036`: {"error":{"code":"HTTP_503","message":{"code":"CLAIM_QA_JSON_PARSE_ERROR","message":"SiliconFlow QA failed after 1 request(s): malformed_json","stage":"LLM_JSON_PARSE","api_request_count":1,"retry_reasons":["malformed_json"],"rate_limit_events":0,"response_audit_path":"/app/artifacts/private/qa-response-audits/q036-full-qa-rerun-q036-1784459283.json"},"request_id":"bbca9d3b-5db4-4016-8805-4c1ca97e311d"}}
- `q037`: {"error":{"code":"HTTP_503","message":{"code":"CLAIM_QA_JSON_PARSE_ERROR","message":"SiliconFlow QA failed after 1 request(s): malformed_json","stage":"LLM_JSON_PARSE","api_request_count":1,"retry_reasons":["malformed_json"],"rate_limit_events":0,"response_audit_path":"/app/artifacts/private/qa-response-audits/q037-full-qa-rerun-q037-1784459396.json"},"request_id":"11addb4c-d751-42f7-a3fe-e597664709df"}}
- `q044`: {"error":{"code":"HTTP_503","message":{"code":"CLAIM_QA_SCHEMA_VALIDATION_ERROR","message":"SiliconFlow QA failed after 1 request(s): schema_validation","stage":"CLAIM_SCHEMA_VALIDATE","api_request_count":1,"retry_reasons":["schema_validation"],"rate_limit_events":0,"response_audit_path":"/app/artifacts/private/qa-response-audits/q044-full-qa-rerun-q044-1784459569.json"},"request_id":"88f605d6-e66c-49e5-b74d-917ac8d99506"}}
