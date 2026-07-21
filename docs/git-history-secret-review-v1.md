# Git History Secret Review v1

- Generated at: `2026-07-20T15:01:30.311252+00:00`
- Total reviewed hits: `36`
- Confirmed real secrets: `0`
- Unresolved hits: `0`
- GIT_HISTORY_SECRET_GATE: `PASSED`
- PUBLIC_RELEASE_ALLOWED: `true`

Only redacted previews are recorded. No full API key, token, cookie, or private key value is written.

## Classification counts

- `DOCUMENTATION_EXAMPLE`: `10`
- `EMPTY_VALUE`: `24`
- `FALSE_POSITIVE`: `1`
- `PLACEHOLDER`: `1`

## Reviewed hits

- `EMPTY_VALUE` `api_key_literal` `46527ea7f086` `.env.example:21` - The variable is intentionally empty or inherited as empty.
- `EMPTY_VALUE` `api_key_literal` `b91e2a661675` `.env.example:27` - The variable is intentionally empty or inherited as empty.
- `EMPTY_VALUE` `api_key_literal` `46527ea7f086` `.env.example:30` - The variable is intentionally empty or inherited as empty.
- `EMPTY_VALUE` `api_key_literal` `b91e2a661675` `.env.example:31` - The variable is intentionally empty or inherited as empty.
- `EMPTY_VALUE` `api_key_literal` `b64783a901d5` `.env.example:34` - The variable is intentionally empty or inherited as empty.
- `EMPTY_VALUE` `api_key_literal` `3a2b6d45e27a` `.env.example:39` - The variable is intentionally empty or inherited as empty.
- `EMPTY_VALUE` `api_key_literal` `46527ea7f086` `.env.example:40` - The variable is intentionally empty or inherited as empty.
- `EMPTY_VALUE` `api_key_literal` `b91e2a661675` `.env.example:41` - The variable is intentionally empty or inherited as empty.
- `EMPTY_VALUE` `api_key_literal` `b64783a901d5` `.env.example:44` - The variable is intentionally empty or inherited as empty.
- `EMPTY_VALUE` `api_key_literal` `a69b9efc6361` `.env.example:49` - The variable is intentionally empty or inherited as empty.
- `PLACEHOLDER` `database_url` `46527ea7f086` `.env.example:5` - The match is in an example/default configuration file and does not contain a real credential.
- `EMPTY_VALUE` `api_key_literal` `6df2d3c99443` `.env.example:55` - The variable is intentionally empty or inherited as empty.
- `EMPTY_VALUE` `api_key_literal` `98058ceef882` `.env.example:58` - The variable is intentionally empty or inherited as empty.
- `EMPTY_VALUE` `api_key_literal` `46527ea7f086` `.env.example:7` - The variable is intentionally empty or inherited as empty.
- `EMPTY_VALUE` `api_key_literal` `3a2b6d45e27a` `.env.example:77` - The variable is intentionally empty or inherited as empty.
- `EMPTY_VALUE` `api_key_literal` `46527ea7f086` `.env.example:86` - The variable is intentionally empty or inherited as empty.
- `FALSE_POSITIVE` `api_key_literal` `46527ea7f086` `data\evaluation\evidence-corpus-v1.jsonl:16649` - The match occurs in evaluation/corpus data and does not match a known live credential format after redaction review.
- `EMPTY_VALUE` `api_key_literal` `46527ea7f086` `docker-compose.yml:16` - The variable is intentionally empty or inherited as empty.
- `EMPTY_VALUE` `api_key_literal` `b91e2a661675` `docker-compose.yml:22` - The variable is intentionally empty or inherited as empty.
- `EMPTY_VALUE` `api_key_literal` `46527ea7f086` `docker-compose.yml:25` - The variable is intentionally empty or inherited as empty.
- `EMPTY_VALUE` `api_key_literal` `b91e2a661675` `docker-compose.yml:26` - The variable is intentionally empty or inherited as empty.
- `EMPTY_VALUE` `api_key_literal` `b64783a901d5` `docker-compose.yml:29` - The variable is intentionally empty or inherited as empty.
- `EMPTY_VALUE` `api_key_literal` `3a2b6d45e27a` `docker-compose.yml:34` - The variable is intentionally empty or inherited as empty.
- `EMPTY_VALUE` `api_key_literal` `46527ea7f086` `docker-compose.yml:35` - The variable is intentionally empty or inherited as empty.
- `EMPTY_VALUE` `api_key_literal` `46527ea7f086` `docs\production-runtime-config-audit.md:32` - The variable is intentionally empty or inherited as empty.
- `DOCUMENTATION_EXAMPLE` `api_key_literal` `46527ea7f086` `docs\stage11a-real-embedding.md:33` - The match is documentation or release-policy text, not a credential.
- `EMPTY_VALUE` `api_key_literal` `46527ea7f086` `docs\stage11b-real-reranker.md:56` - The variable is intentionally empty or inherited as empty.
- `DOCUMENTATION_EXAMPLE` `auth_header` `46527ea7f086` `scripts\audit_evidence_qa_dev_v3_1.py:111` - The script contains detector/test strings for security checks, not a committed credential.
- `DOCUMENTATION_EXAMPLE` `api_key_literal` `46527ea7f086` `scripts\audit_stage13_checkpoint_v1.py:87` - The script checks whether a line starts with an API-key variable name; no literal secret value is present.
- `DOCUMENTATION_EXAMPLE` `auth_header` `4abcf22cc507` `scripts\prepare_stage13_9_citation_recall_audit_v1.py:329` - The script contains detector/test strings for security checks, not a committed credential.
- `DOCUMENTATION_EXAMPLE` `auth_header` `46527ea7f086` `scripts\prepare_stage13_9_citation_recall_audit_v1.py:346` - The script contains detector/test strings for security checks, not a committed credential.
- `DOCUMENTATION_EXAMPLE` `auth_header` `46527ea7f086` `tests\test_stage11a_embedding.py:78` - The match is a test fixture or assertion for redaction/security behavior.
- `DOCUMENTATION_EXAMPLE` `auth_header` `46527ea7f086` `tests\test_stage11b_reranker.py:85` - The match is a test fixture or assertion for redaction/security behavior.
- `DOCUMENTATION_EXAMPLE` `auth_header` `46527ea7f086` `tests\test_stage11c_qa.py:102` - The match is a test fixture or assertion for redaction/security behavior.
- `DOCUMENTATION_EXAMPLE` `auth_header` `98058ceef882` `tests\test_stage11c_qa.py:85` - The match is a test fixture or assertion for redaction/security behavior.
- `DOCUMENTATION_EXAMPLE` `auth_header` `3a2b6d45e27a` `tests\test_stage11c_qa.py:98` - The match is a test fixture or assertion for redaction/security behavior.
