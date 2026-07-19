# Production Runtime Configuration Audit

Generated on: 2026-07-18  
Branch: `eval/retrieval-recall-benchmark-v1`

## Result

Status: **PARTIAL**

The original host/container mismatch has been fixed at the configuration-injection level. The rebuilt API container now starts with the configured Production LLM provider/model and no Production configuration issues.

The Production LLM provider preflight has now **passed** with a real minimal chat completion against the configured SiliconFlow provider/model. The capability endpoint still reports runtime capability as `configured` rather than `verified` because it does not itself execute an external provider preflight on every request.

Full QA, Production Deep Research, Production Soak, and `v1.0.0-portfolio` release remain prohibited because retrieval validation/holdout/generalization gates are still failing.

## Configuration precedence

Observed implementation:

1. Explicit process environment variables.
2. Docker Compose `environment:` entries, interpolated from the Compose caller environment and `.env`.
3. Pydantic Settings `.env` file for host-side local runs.
4. Code defaults in `src/paper_research/config.py`.

Docker Compose does not use `env_file`; instead `docker-compose.yml` interpolates values directly:

- `APP_PROFILE: ${APP_PROFILE:-baseline}`
- `EMBEDDING_PROVIDER: ${EMBEDDING_PROVIDER:-hash}`
- `LLM_PROVIDER: ${LLM_PROVIDER:-template}`
- `LLM_MODEL: ${LLM_MODEL:-template-v1}`
- `LLM_BASE_URL: ${LLM_BASE_URL:-}`
- `LLM_API_KEY: ${LLM_API_KEY:-}`
- `CHECKPOINT_PROVIDER: postgres`
- `CHECKPOINT_DATABASE_URL: postgresql://paper:paper@postgres:5432/paper_research`

Therefore, `docker compose restart` is not sufficient after environment changes; the API container must be force-recreated.

## Host `.env` safe summary

Secrets are not printed. API-key values were checked only by presence, length, and short SHA-256 prefix.

| Variable | Host state |
|---|---|
| `APP_PROFILE` | present; value `production` |
| `LLM_PROVIDER` | present; value `siliconflow` |
| `LLM_MODEL` | present; value `Qwen/Qwen3-8B` |
| `LLM_BASE_URL` | present; host `api.siliconflow.cn` |
| `LLM_API_KEY` | present; length 51; SHA-256 prefix recorded in command output only |
| `LLM_TEMPERATURE` | present; value `0` |
| `EMBEDDING_PROVIDER` | present; value `jina` |
| `EMBEDDING_MODEL` | present; value `jina-embeddings-v5-text-small` |
| `RERANK_PROVIDER` | present; value `lexical` |
| `RERANK_MODEL` | present; value `lexical-v1` |
| `RERANK_ENABLED` | present; value `false` |
| `CHECKPOINT_PROVIDER` | present; host value `memory` |

The host checkpoint provider differs from Compose by design: Compose forces `CHECKPOINT_PROVIDER=postgres` for the API container.

## Initial running API capability evidence

`GET http://localhost/api/v1/capabilities` returned:

- `overall=degraded`
- `profile=production`
- `embedding=jina/jina-embeddings-v5-text-small`
- `reranker=disabled`
- `llm=template/template-v1`
- `production_configuration_issues=["LLM_API_KEY", "LLM_BASE_URL", "LLM_PROVIDER"]`
- `langgraph_checkpoint=postgres`
- OCR/Tesseract available
- Redis available and used

This meant the running API container could not be treated as a verified Production LLM runtime.

## Root cause assessment

Two issues were observed:

1. The API container had previously been created with stale/default LLM settings, so the container reported Template LLM while the host `.env` had SiliconFlow configured.
2. After the user rebuilt and force-recreated the container, the API crashed on startup because Compose passed empty strings for optional numeric fields:
   - `LLM_INPUT_COST_PER_MILLION`
   - `LLM_OUTPUT_COST_PER_MILLION`

Pydantic attempted to parse those empty strings as floats. The code now treats empty strings for optional numeric settings as `None`.

Secondary blocker: the current Codex sandbox user still cannot read Docker Desktop context metadata directly, so daemon-level `docker version/info` remains partially blocked in Codex even though project-level Compose operations work.

## Fix applied

Code:

- `src/paper_research/config.py`
- `tests/test_stage10_profiles_and_review.py`

Validation:

- `tests/test_stage10_profiles_and_review.py` covers empty optional numeric env values parsing to `None`.
- `docker compose build --no-cache api` rebuilt the image successfully.
- The Docker build produced `paper-research-agent-0.9.0rc3`.
- `docker compose up -d --force-recreate api nginx` completed successfully after the fix.
- `docker compose ps` shows `research-api-1` healthy.

Post-fix API evidence:

- `GET /api/v1/health`
  - `status=healthy`
  - `version=0.9.0rc3`
  - `display_version=0.9.0-rc3`
  - PostgreSQL/Qdrant/Redis up
- `GET /api/v1/capabilities`
  - `profile=production`
  - `llm=siliconflow/Qwen/Qwen3-8B`
  - `llm.status=configured`
  - `llm.verified=false`
  - `production_configuration_issues=[]`
  - `langgraph_checkpoint=postgres`
  - OCR/Tesseract available
  - Redis used

## Provider preflight evidence

Command:

```powershell
.\.venv\Scripts\python.exe scripts\check_llm_provider_health_v1.py --require-minimal-completion --output data\evaluation\provider-health-v1.json
```

Result file:

- `data/evaluation/provider-health-v1.json`

Recorded safe output:

- `status=PASSED`
- `safe_to_start_batch=true`
- `base_url_host=api.siliconflow.cn`
- `dns_status=passed`
- `tcp_status=passed`
- `tls_status=passed`
- `models_endpoint_status=passed`
- `minimal_completion_status=passed`
- `minimal_completion_json_valid=true`
- `minimal_completion_model=Qwen/Qwen3-8B`
- `factory_provider=siliconflow`
- `factory_model=Qwen/Qwen3-8B`
- `template_fallback=false`
- `prompt_tokens=16`
- `completion_tokens=323`
- `total_tokens=339`
- `minimal_completion_ms=14815.402`
- `api_key_recorded=false`
- `authorization_header_recorded=false`

No API Key or Authorization header was printed or persisted by the preflight output.

## Remaining required remediation

Run from a Docker-authorized desktop PowerShell session:

```powershell
cd D:\Agents\Codex\research
docker compose config
docker compose build --no-cache api
docker compose up -d --force-recreate api nginx
docker compose ps
```

The explicit provider preflight has passed. Do not run Full QA or Deep Research until retrieval generalization gates pass.

Health verification commands:

```powershell
Invoke-RestMethod http://localhost/api/v1/health | ConvertTo-Json -Depth 8
Invoke-RestMethod http://localhost/api/v1/capabilities | ConvertTo-Json -Depth 8
```

Production LLM provider preflight is passed. A full QA/Deep Research trace with the configured provider/model and no Template fallback is still intentionally not run because retrieval generalization gates remain blocked.

## Release decision

Configuration mismatch is fixed and provider preflight passed, but release readiness remains **PARTIAL** because retrieval generalization gates are still failing. No Full QA, Production Deep Research, Production Soak, or `v1.0.0-portfolio` release is authorized.
