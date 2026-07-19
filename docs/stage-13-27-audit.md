# Stage 13.27 Portfolio Gate Audit

Generated on: 2026-07-18  
Branch: `eval/retrieval-recall-benchmark-v1`  
HEAD: `98058ceef882b317afa9b6f2086b9da9ffdac3d0`  
Purpose: audit the real current state for `v1.0.0-portfolio` without inventing human review, model calls, metrics, token usage, cost, deployment state, or release status.

## Decision

Current conclusion: **C. Only partially satisfies the portfolio gates; release is prohibited.**

The project remains a portfolio/release-candidate-quality system, not `v1.0.0-portfolio`. The strongest current blockers are retrieval/citation quality, Dev/holdout generalization evidence, production container LLM configuration mismatch, insufficient long soak duration, and missing minimum public-release security/content-claims audits.

Do not create a `v1.0.0-portfolio` tag, do not update package version to `1.0.0`, and do not describe the system as Production-ready until every gate below has reproducible evidence.

## Current repository and runtime state

| Item | Status | Evidence | Blocker / note | Next action | User/key needed |
|---|---:|---|---|---|---|
| Git branch | PASS | `git branch --show-current` -> `eval/retrieval-recall-benchmark-v1` | None | Keep branch isolated. | No |
| Git HEAD | PASS | `git rev-parse HEAD` -> `98058ceef882b317afa9b6f2086b9da9ffdac3d0` | None | Do not merge main unless requested. | No |
| Tag at HEAD | PASS | `git tag --points-at HEAD` returned empty | No release tag created. | Do not create v1 tag. | Yes, for any tag |
| Worktree | PARTIAL | `git status --short` shows only two untracked review ZIP files before this audit doc | Local-only review packages remain untracked. | Keep ZIPs uncommitted. | No |
| Package version | PARTIAL | `pyproject.toml` has `version = "0.9.0rc3"`; `src/paper_research/__init__.py` has `0.9.0-rc1` | Version metadata is not `1.0.0-portfolio` and is internally inconsistent. | Align only after all portfolio gates pass. | No |
| README release claim | PASS | `README.md` says current tree is candidate for `v0.9.0-rc3`, not `v1.0.0`, not Production-ready | Honest, conservative wording retained. | Continue content audit before public release. | No |
| Docker services | PARTIAL | `docker compose ps` shows API, Nginx, PostgreSQL, Qdrant, Redis up; API/Postgres/Redis healthy | `docker version` cannot connect to Docker API for the current user: permission denied on Docker pipe/config. | Resolve Docker user permission or run from approved shell context; re-run daemon-level audit. | Possibly user/admin |
| API health | PASS | `GET http://localhost/api/v1/health` -> `healthy`; postgres/qdrant/redis up | Runtime health is up. | Keep for smoke evidence only. | No |
| API capabilities | PARTIAL | `GET http://localhost/api/v1/capabilities` -> overall `degraded`; OCR/Tesseract available; Redis used; reranker disabled | Container reports `llm=template/template-v1` and production issues `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_PROVIDER`, while host `.env` has SiliconFlow configured. | Recreate/restart API with intended Production LLM env before any final Production acceptance. | Yes if key/env not injected |

## Twelve `v1.0.0-portfolio` hard gates

| # | Gate | Status | Supporting evidence | Blocking reason | Next step | User/key needed |
|---:|---|---:|---|---|---|---|
| 1 | 50 evaluation items fully human-processed; formal metrics count only `approved` | PASS | `data/evaluation/gold-set-v1.jsonl`: 50 rows, 50 `review_status=approved`, 48 answerable, 2 unanswerable. `data/evaluation/retrieval-gold-v2.jsonl`: 50 signed, 2 query revisions approved and 48 not-required. | No current blocker for dev gold. This is still internal/dev gold, not a public benchmark. | Preserve audit trail; create `gold-test-v1` only by explicit human process if needed. | No |
| 2 | At least one real Embedding successfully ran | PASS | `docs/stage11a-real-embedding.md`; `data/evaluation/reranker-ablation-v1.json` records `jina-embeddings-v5-text-small`, 1024 dimensions, 34 papers, 2062 points in `papers_jina_eval34_v2__20260713152149`. Qdrant collection query confirms 2062 points and vector size 1024. | None for engineering proof. Not enough by itself for v1 quality. | Keep Jina collection/version fixed; do not mix with Hash vectors. | No, unless rebuilding |
| 3 | At least one real LLM completed QA and Deep Research | PARTIAL | QA: `docs/qa-production-v1.md` records SiliconFlow `Qwen/Qwen3-8B`, prompt `qa-production-v1`, reranker disabled. Deep Research smoke: `data/evaluation/deep-research-smoke-v1.json` selects q003/q005/q049 successful engineering runs. | Deep Research summary is explicitly `ENGINEERING_ONLY_EXPLICIT_SUMMARY`, `quality_gate=NOT_EVALUATED`; container capabilities currently show Template LLM and Production LLM config issues. | Re-run/verify Production container with real LLM env before final QA/Deep Research acceptance; run quality evaluation only after retrieval/citation gates recover. | Yes if API key/env missing |
| 4 | Cross-Encoder compared; may remain disabled after evaluation | PASS | `data/evaluation/reranker-ablation-v1.json`: model `jina-reranker-v3`, no LLM/Deep Research, decision `recommend_rerank_enabled=false`; failures/fallbacks zero; RERANK remains disabled. | No blocker; negative result is valid. | Keep `RERANK_ENABLED=false` unless a future approved-gold comparison passes. | No |
| 5 | Claim-level citation validation passed | PARTIAL | Claim-level machinery and audits exist: `data/evaluation/claim-gold-citation-comparison-v1.json`, `docs/claim-gold-citation-comparison-v1.md`, Stage 13.27 benchmark. | Quality is not passing: v1 release gates record exact citation precision `0.103009`, citation recall `0.096875`, unsupported claims unacceptable; Stage 13.27 says validation/holdout failed and generalization evidence insufficient. | Continue retrieval/evidence selection work; do not enter human citation audit until candidate is stable. | No for offline; yes for future live |
| 6 | PostgreSQL Checkpoint recovery continues to pass | PARTIAL | `docs/langgraph-recovery-audit.md` says Stage 10 recovery passed. `data/evaluation/deep-research-smoke-v1.json` has persisted runs/attempts. API capabilities show `langgraph_checkpoint` available via postgres. | Host settings currently show `checkpoint_provider=memory`; v1 release gates still include `DR-07` blocked for successful resumed completion after provider failure. | Reconcile host/container checkpoint provider and repeat final Production stop/resume only when authorized. | Possibly yes for live LLM |
| 7 | Redis continues to have real usage | PASS | `GET /health`: `used=True; keys=3; cache_hit_rate=0.333333`. `GET /capabilities`: Redis available, used, TTL 3600, key_count 3. `docker compose exec redis redis-cli DBSIZE` -> 3. Code uses Redis for cache, rate limit, import lock. | No immediate blocker for “actual use,” but final production audit doc requested as `docs/redis-production-audit.md` is absent. | Generate dedicated Redis production audit with repeat cache/TTL/degrade checks. | No |
| 8 | Docker OCR remains available | PASS | `docs/ocr-audit-v1.md` says rebuilt Docker image has Tesseract 5.5.0 and English data. API capabilities show PyMuPDF available, OCR engine available, Tesseract `/usr/bin/tesseract`. | Full final smoke with real Embedding + real LLM over three PDF classes has not been re-run in this Stage 13.27 audit. | Re-run Docker OCR e2e only after Production LLM env is aligned. | Possibly yes for model key |
| 9 | PostgreSQL and Qdrant restore continue to pass | PARTIAL | `docs/backup-restore-audit.md` records Stage 10 PASS. Qdrant current active collection is green with 2062 points. | The requested final revalidation against current Production collection/database has not been re-executed in this audit. | Re-run pg_dump/restore and Qdrant snapshot restore before final release. | No, but needs Docker permission |
| 10 | Longer Soak Test completed | BLOCKED | `artifacts/soak-test-v1.json`: duration `182.703` seconds, profile `baseline`, 276 queries, 4 imports, 3 Deep Research runs, failures 0. | Required portfolio soak is suggested 2 hours (or 4 hours if possible), with real LLM/Production-like workload. Current evidence is short and baseline. | Run `soak-test-portfolio-v1` for >=2 hours after Production LLM and retrieval gates are ready. | Yes for live model cost/time |
| 11 | Minimum security audit complete | BLOCKED | No `docs/security-audit-v1.md`; no `docs/public-demo-security.md` found. `v1-release-gates.json` records Qdrant HTTP/API-key warning and client/server compatibility warning as v1 blockers. | Minimum public-release security closure is missing. | Run/document secrets, upload, URL, API, container audit without printing secrets. | Possibly user/admin for Docker/history scans |
| 12 | README/demo/resume contain no exaggerated claims | PARTIAL | README is conservative; however `docs/demo-video-script.md`, `docs/interview-guide.md`, and `docs/resume-description.md` still contain strong metrics such as Dense Hit@1 0.80 / Hybrid Recall@10 0.90. No `docs/content-claims-audit-v1.md` exists. | Quantified content claims need traceable mapping to approved data/model/index/prompt/run, and stale claims must be reconciled with newer Stage 13.27 failures. | Produce `docs/content-claims-audit-v1.md` and edit public materials if needed. | No |

## Additional Stage 13.27 audit findings

### Gold data

| File | Rows | Status counts | SHA-256 |
|---|---:|---|---|
| `data/evaluation/gold-set-v1.jsonl` | 50 | `review_status=approved`: 50; answerable true/false: 48/2 | `24b21d7ce5264d4f22230cfb6bc9ec788ef6b76dc0ad629a20ae682c5184599e` |
| `data/evaluation/retrieval-gold-v2.jsonl` | 50 | `review_status=approved`: 50; query revision approved/not-required: 2/48 | `a196fc0c40823dd66b3972cf1d455d647325a20872cfe1f81685b967ec4e2e8d` |
| `data/evaluation/claim-evidence-gold-dev-v1.jsonl` | 27 | `adjudication_status=approved`: 27 | `e1aadcae82fb8a7f867eb600ffb0b2836813fd70e38c70502992cc7e4faa4bcd` |

No script or LLM approval was performed in this audit.

### Current provider/profile state

Host `.env` as loaded by `Settings`:

- `APP_PROFILE=production`
- Active Qdrant collection: `papers_jina_eval34_v2__20260713152149`
- Embedding: `jina / jina-embeddings-v5-text-small`, 1024 dimensions
- LLM: `siliconflow / Qwen/Qwen3-8B`, billing mode `free`
- Reranker: `lexical / lexical-v1`, `RERANK_ENABLED=false`
- Redis: configured at localhost
- Checkpoint provider: `memory`
- Production configuration issues on host settings: none

Running Docker API `/capabilities`:

- Overall: `degraded`
- Embedding capability: `jina/jina-embeddings-v5-text-small`, configured, not verified in that response
- LLM capability: `template/template-v1`
- Production configuration issues: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_PROVIDER`
- LangGraph checkpoint: postgres available/verified

This host/container mismatch must be resolved before any final Production acceptance.

### Stage 13.27 retrieval benchmark state

Evidence: `data/evaluation/retrieval-recall-benchmark-v1-final-audit.json`

- `RETRIEVAL_BENCHMARK_V1_ENGINEERING_GATE=PASSED`
- `RETRIEVAL_BENCHMARK_SAMPLE_SIZE_SUFFICIENT=false`
- `RETRIEVAL_SPLIT_LEAKAGE_GATE=PASSED`
- `HYBRID_RETRIEVAL_V1_ENGINEERING_GATE=PASSED`
- `HYBRID_RETRIEVAL_V1_DEV_GATE=PASSED`
- `HYBRID_RETRIEVAL_V1_VALIDATION_GATE=FAILED`
- `HYBRID_RETRIEVAL_V1_HOLDOUT_GATE=FAILED`
- `RETRIEVAL_GENERALIZATION_EVIDENCE=INSUFFICIENT`
- `END_TO_END_COMPATIBILITY_REPLAY=NOT_AUTHORIZED`
- `NEXT_LIVE_READY=false`
- `READY_FOR_FULL_QA=false`
- `live_llm_executed=false`
- `external_embedding_api_executed=false`
- `external_reranker_executed=false`
- `deep_research_executed=false`

This explicitly blocks the next live/full-QA path.

### Reranker state

Evidence: `data/evaluation/reranker-ablation-v1.json`

- Cross-Encoder: `jina-reranker-v3`
- Corpus/collection: 34 papers, 2062 points, Jina 1024d
- LLM called: false
- Deep Research called: false
- Recommendation: keep `RERANK_ENABLED=false`
- Reason: Hit@1/MRR did not improve and P95 total latency did not satisfy all acceptance conditions, despite zero failures/fallbacks.

### Deep Research state

Evidence: `data/evaluation/deep-research-smoke-v1.json`

- Status: `ENGINEERING_ONLY_EXPLICIT_SUMMARY`
- Quality gate: `NOT_EVALUATED`
- Selected runs: `live-q003-798ac68288e0`, `live-q005-03f669606bb7`, `live-q049-4c11db9a2c1d`
- Failed attempts retained: 3

This is valid engineering smoke evidence, not representative quality evidence.

### Soak state

Evidence: `artifacts/soak-test-v1.json`

- Duration: 182.703 seconds
- Profile: baseline
- Queries: 276
- Periodic imports: 4
- Deep Research runs: 3
- API restart performed: true
- Failures: 0
- Tokens/cost: 0 / 0 USD

This is not a 2-hour or 4-hour Production-like portfolio soak.

## Commands actually executed for this audit

No real LLM, external embedding API, external reranker, Deep Research, Full QA, or release/tag action was executed.

Commands:

```powershell
git status --short
git branch --show-current
git rev-parse HEAD
git tag --points-at HEAD
Get-ChildItem -Name
Get-ChildItem ... | Select-String ...
.\.venv\Scripts\python.exe - <read-only audit snippets>
docker version
docker compose ps
.\.venv\Scripts\python.exe scripts\run_evaluation_production_v1.py --help
git restore --source=HEAD -- data/evaluation/results-production-v1.csv data/evaluation/results-production-v1.json docs/evaluation-report-production-v1.md
Invoke-RestMethod -Uri http://localhost/api/v1/health -TimeoutSec 5
Invoke-RestMethod -Uri http://localhost/api/v1/capabilities -TimeoutSec 5
docker compose exec -T redis redis-cli DBSIZE
Invoke-RestMethod -Uri http://localhost:6333/collections/papers_jina_eval34_v2__20260713152149 -TimeoutSec 5
```

Note: `scripts/run_evaluation_production_v1.py --help` does not implement a help-only path and refreshed three evaluation files. Those changes were immediately reverted with `git restore`; no live model call or Deep Research was triggered by that command.

## Remaining blockers

| Blocker | Reason | Needs user/key? | Next executable action |
|---|---|---:|---|
| Production container LLM mismatch | Docker API capabilities still report Template LLM and missing LLM production config, while host settings report SiliconFlow configured. | Yes if container env/key must be supplied | Recreate/restart API with intended `.env` and re-check `/capabilities`. |
| Stage 13.27 validation/holdout failed | Latest retrieval benchmark says generalization evidence is insufficient and `NEXT_LIVE_READY=false`. | No for offline analysis | Continue targeted obligation retrieval or evidence selection only after deciding next offline fix. |
| Citation quality below v1 threshold | v1 gates record exact citation precision/recall and unsupported claim rate failing. | No for offline; yes for future live validation | Improve retrieval/evidence selection; rerun controlled candidate evaluation only after offline gates pass. |
| Deep Research quality not evaluated | Current Deep Research evidence is engineering-only smoke. | Yes for live model | Run representative Deep Research quality only after retrieval/citation gates pass. |
| Successful resumed completion blocker | v1 gates still mark `DR-07` blocked after provider failure. | Yes for live provider | Repeat one controlled stop/resume when provider stability is approved. |
| Long soak missing | Existing soak is 182.703 seconds and baseline. | Yes for live model/time | Run `soak-test-portfolio-v1` for >=2 hours in Production-like mode. |
| Security audit missing | `docs/security-audit-v1.md` and `docs/public-demo-security.md` absent. | Possibly for Docker/history access | Run minimum public-release security audit without printing secrets. |
| Content-claims audit missing | `docs/content-claims-audit-v1.md` absent; public docs contain strong metrics requiring traceability checks. | No | Audit README/demo/resume/interview/release claims and edit overclaims. |
| Version not v1 | Package remains `0.9.0rc3`; runtime `__version__` remains `0.9.0-rc1`. | No | Align versions only after every hard gate passes. |
| Docker daemon permission | `docker version` cannot access Docker API for current user, though `docker compose ps` works. | Possibly user/admin | Fix Docker Desktop/pipe permission before final daemon-level audit. |

## Final classification

`C. Only partially satisfies Portfolio gates; release is prohibited.`

The project has strong engineering evidence for many components, but it is **not** `v1.0.0-portfolio`, **not** Production-ready, and should remain below final release status until the blockers above are resolved with reproducible evidence.
