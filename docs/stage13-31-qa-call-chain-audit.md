# Stage 13.31 QA Call Chain Audit

| Stage | Status after fix | Failure code if blocked |
|---|---|---|
| `QA_REQUEST_VALIDATION` | passed for q002 smoke | `CLAIM_QA_REQUEST_VALIDATION_ERROR` |
| `QA_CONTEXT_BUILD` | completed, but context missed gold evidence | `BLOCKED_BY_CONTEXT_DATA` |
| `LLM_REQUEST_BUILD` | fixed; prompt version now supported | `CLAIM_QA_CONFIGURATION_ERROR` |
| `LLM_PROVIDER_CALL` | passed; real SiliconFlow request completed | `CLAIM_QA_PROVIDER_ERROR` |
| `LLM_RESPONSE_EXTRACT` | passed for observed public answer envelope | `CLAIM_QA_PROVIDER_RESPONSE_ERROR` |
| `LLM_JSON_PARSE` | no failure observed in q002 smoke | `CLAIM_QA_JSON_PARSE_ERROR` |
| `CLAIM_SCHEMA_VALIDATE` | passed; model returned valid refusal shape | `CLAIM_QA_SCHEMA_VALIDATION_ERROR` |
| `CLAIM_REFERENCE_RESOLVE` | not applicable because claims were empty | `CLAIM_QA_REFERENCE_RESOLUTION_ERROR` |
| `CLAIM_CITATION_VALIDATE` | passed vacuously for refusal; no citations | `CLAIM_QA_CITATION_VALIDATION_ERROR` |
| `QA_RESPONSE_BUILD` | passed | `CLAIM_QA_RESPONSE_BUILD_ERROR` |

The API no longer exposes raw `ValueError` for unsupported prompt versions. It returns
stable code/stage metadata and avoids raw tracebacks, local paths, API keys, headers, and
full model responses.

Current q002 failure is not classified as provider incompatibility. It is classified as
`BLOCKED_BY_CONTEXT_DATA` because the retrieved context did not contain the gold answer
evidence required for an answerable QA item.
