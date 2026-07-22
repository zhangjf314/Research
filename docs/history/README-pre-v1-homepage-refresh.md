# PaperResearch Agent

## Stage 13.40 portfolio release status

Current conclusion: **A. All local `v1.0.0-portfolio` hard gates passed; merge,
tag, push, and remote release still require explicit user authorization.**

- Current package/runtime version is `1.0.0+portfolio` / display
  `1.0.0-portfolio` in the local release-preparation tree. Merge, tag, push,
  and remote release still require explicit user authorization.
- DeepSeek `deepseek-v4-flash` completed the 50-record internal Full QA
  engineering gate with 50/50 completed, 0 failures, no template fallback, and
  strict citation identifier/context/page validation.
- One bounded Deep Research run completed with strict citation validation.
- Git-history secret review, PostgreSQL checkpoint recovery, PostgreSQL
  backup/restore, Qdrant snapshot/restore, Docker OCR roundtrip, and the
  Portfolio 30-minute stability test all passed in Stage 13.40.
- Strong semantic grounding and strong generalization claims are still disabled:
  `SEMANTIC_CLAIM_SUPPORT_AUDIT=NOT_FORMALLY_VALIDATED` and
  `STRONG_GENERALIZATION_CLAIM_ALLOWED=false`.

See [`docs/portfolio-release-audit-v1.md`](docs/portfolio-release-audit-v1.md)
and [`docs/release-checklist-v1.0.0-portfolio.md`](docs/release-checklist-v1.0.0-portfolio.md).

## Portfolio evaluation status

The current Portfolio evaluation policy uses three dataset tiers:

- `gold-dev-v1`: 50 human-approved records used as an internal development
  evaluation set for retrieval, reranker, QA, and claim-citation evaluation.
- `retrieval-diagnostic-v1`: 27 claim-level diagnostic records used for failure
  analysis and regression checks. This data is not blind.
- `shadow-holdout-pilot-v1`: optional 10-15-sample blind pilot; not required for
  Portfolio Full QA and not yet created.

Allowed public wording:

> 基于 50 条人工审核的内部评测数据完成检索和问答评测

The project does not claim strict generalization, production-grade
generalization, or results on a large independent blind benchmark. Current
retrieval evidence is `DIAGNOSTIC_ONLY`, with
`STRONG_GENERALIZATION_CLAIM_ALLOWED=false`.

## Release readiness status (Stage 12)

The highest published version is `v0.9.0-rc2`. The current tree is a candidate for
`v0.9.0-rc3` after the strict RC gate and human release review; it does **not** satisfy
`v1.0.0` and is **not Production-ready**.
Package metadata is staged as `0.9.0rc3`, but Stage 12 does not create a Tag or commit.

- The real Embedding path uses `jina-embeddings-v5-text-small` on the versioned
  34-document, 2,062-point corpus.
- `RERANK_ENABLED=false` remains the required default because `jina-reranker-v3` did not
  pass the quality/latency enablement gate.
- Real QA with SiliconFlow `Qwen/Qwen3-8B` passed structured JSON/schema and identifier
  validation, but answer and citation quality remain below the Production thresholds.
- Deep Research has passed only the bounded engineering smoke, budget, isolation, checkpoint,
  resume-idempotency, and failure-accounting checks. Its quality evaluation is blocked, and a
  successful end-to-end completion after resume is not verified.

Run the single auditable release check without making model calls:

```powershell
.\.venv\Scripts\python.exe scripts\check_release_readiness_v1.py --target rc --strict
.\.venv\Scripts\python.exe scripts\check_release_readiness_v1.py --target v1 --strict
```

The strict v1 command is expected to return nonzero while quality, resumed-completion, and
operations gates remain open. See [`docs/v1-release-gates.md`](docs/v1-release-gates.md),
[`docs/known-issues-v1.md`](docs/known-issues-v1.md), and
[`docs/v1-gap-closure-plan.md`](docs/v1-gap-closure-plan.md).

## Stage 11A — real Embedding retrieval evaluation

Work on branch `eval/real-embedding-v1` adds a Jina
`jina-embeddings-v5-text-small` provider and a pure-retrieval four-way ablation runner.
The Stage 11A path keeps `RERANK_ENABLED=false`, uses `LLM_PROVIDER=template`, and does
not call QA or Deep Research. Hash and Jina vectors use independent versioned Qdrant
collections; a failed Production build never switches the logical collection.

Offline checks:

```powershell
python scripts\check_embedding_provider.py
python -m pytest
python -m ruff check .
python -m compileall src scripts
```

After configuring the ignored local Production environment, the real workflow is:

```powershell
Copy-Item .env.stage11a.local .env -Force
python scripts\check_embedding_provider.py
Invoke-RestMethod -Method Post http://localhost/api/v1/indexes/rebuild
python scripts\run_retrieval_ablation_v1.py
```

Never paste or log `EMBEDDING_API_KEY`. See
[`docs/stage11a-real-embedding.md`](docs/stage11a-real-embedding.md) for configuration,
collection, metric, and rollback boundaries.

Stage 11A.5 corrects the retrieval protocol without enabling a Reranker or LLM. It
preserves `gold-set-v1.jsonl`, records the 34-document Production boundary in
`data/evaluation/production-corpus-v1.json`, and evaluates known-paper, multi-paper,
and unanswerable scopes separately. See
[`docs/retrieval-gold-v2-audit.md`](docs/retrieval-gold-v2-audit.md) and
[`docs/retrieval-ablation-v2.md`](docs/retrieval-ablation-v2.md). The two rewritten
unanswerable queries were human-approved by `zjf` on 2026-07-13; Stage 11B may now begin
with Reranker still disabled by default.

Stage 11B adds a strict Jina `jina-reranker-v3` adapter and a reranker-only ablation
runner. Retrieval is frozen to the Stage 11A.5 Jina Structural Hybrid snapshot:
retrieve Top-30, rerank all Top-30 candidates, and evaluate Top-10. Production
configuration fails explicitly when the model or API key is missing; formal ablations
require `RERANK_ALLOW_FALLBACK=false`. The real three-way run completed on 2026-07-14
with zero failures and zero fallbacks. Jina v3 did not improve paper-scoped Hit@1 or
MRR and its total P95 latency was 63.66 seconds, so `RERANK_ENABLED=false` remains the
evidence-based default. See
[`docs/reranker-ablation-v1.md`](docs/reranker-ablation-v1.md) and
[`docs/stage11b-real-reranker.md`](docs/stage11b-real-reranker.md).

Stage 11C adds an evidence-bound SiliconFlow `Qwen/Qwen3-8B` ordinary-QA adapter,
strict claim-level paper/page/block citations, token-budget context tracing, and a
resumable smoke/dev/full evaluator. Retrieval remains Jina Structural Hybrid and the
Reranker remains disabled. The real 50-query run completed with 100% final JSON/schema
validity and 100% refusal accuracy, but only 38.9% required-claim coverage, 10.3%
citation precision, and 9.7% citation recall. These results validate the integration,
not production answer quality; see
[`docs/qa-production-v1.md`](docs/qa-production-v1.md) and
[`docs/stage11c-qa-audit.md`](docs/stage11c-qa-audit.md).

Stage 11C.5 isolates retrieval, context distraction, and Gold-citation strictness without
changing the frozen model or retrieval protocol. Exact Gold evidence was present in 43.8%
of retrieved contexts; a Gold-only Oracle raised answerable accuracy from 87.5% to 95.8%
and required-claim coverage from 38.9% to 54.2%. Adding distractors reduced exact citation
precision from 95.8% to 81.5%, while simply appending missing Gold recovered 5/6 prior
answerable refusals but left a 56.1% strict unsupported rate. Oracle rows are diagnostic,
not Production metrics. The evidence supports improving retrieval/context selection before
Deep Research or a larger model; see
[`docs/qa-context-diagnostics-v1.md`](docs/qa-context-diagnostics-v1.md).

Stage 11C.6 evaluates bounded retrieval/context changes while keeping Jina Embedding,
SiliconFlow `Qwen/Qwen3-8B`, `qa-production-v1`, chunks, queries, filters, and Gold frozen;
the Reranker remains disabled. The best pure-retrieval candidate uses no structural expansion,
caps each page at two structural chunks, and weights Dense/Lexical RRF at 0.7/0.3. It reached
52.1% exact-Gold availability and 75.0% Gold-page availability, but the 46 completed QA rows
only raised exact citation precision from 10.3% to 11.0%, reduced citation recall to 5.9%, and
increased unsupported rate to 87.7%; q033 and q044 repeatedly failed strict page-citation
validation. The 30-item human citation audit is still pending. Stage 11C.6 therefore does not
authorize Stage 11D smoke; see
[`docs/retrieval-context-optimization-v1.md`](docs/retrieval-context-optimization-v1.md).

Stage 11C.7 accepts a completed **AI-assisted manual citation audit, 30-sample
stratified review**; it is not an independent blind review or a full-dataset human
precision estimate. Strict human support is 5/30 (16.7%) and lenient support is 7/30
(23.3%). Token-set semantic support has only 30% strict and 40% lenient precision in
this failure-enriched sample, so the earlier 81.6% value remains a lexical diagnostic,
not citation correctness. The q033/q044 failures were traced to inconsistent block/page
serialization and uninformative retries; exact block-page maps plus bounded legal-triple
retry guidance fixed both in one retry while strict validation continued rejecting the
first illegal outputs. At most three engineering-only Stage 11D smoke cases may proceed
under explicit request/token/cost/round limits; Stage 11D quality evaluation remains
blocked. See [`docs/stage11c7-citation-audit-v1.md`](docs/stage11c7-citation-audit-v1.md).

> Stage 10 acceptance status (2026-07-13): the repository remains `v0.9.0-rc1`,
> not `v1.0.0`. Baseline is reproducible; Production is intentionally blocked until
> real model credentials and 50/50 human-approved gold records are available.

## Stage 10 profiles and verified capabilities

| Area | Baseline (`APP_PROFILE=baseline`) | Production (`APP_PROFILE=production`) |
|---|---|---|
| Embedding | Hash `hash-v1`, collection `papers_hash_v1` | OpenAI-compatible embedding; configuration required |
| Reranker | Disabled by default; lexical retained for ablation | Cross-encoder adapter; acceptance pending |
| Generator | Deterministic template | OpenAI-compatible structured Claim JSON; configuration required |
| CI/offline | Supported | Fails explicitly when required providers are missing |
| Formal metrics | RC provisional results only | Blocked: 0/50 human-approved records |

Verified in Docker during Stage 10:

- Redis is used for external-search TTL caching and API rate limiting; health reports
  usage and hit rate. Semantic Scholar returned 429 and arXiv results provided the
  controlled fallback; the repeated query hit Redis.
- LangGraph uses PostgreSQL checkpoints in Compose. A paused fixed `thread_id` resumed
  after the API image/container was replaced, without rerunning committed nodes.
- Tesseract 5.5 is installed in the image. Text, mixed, and fully scanned PDFs completed
  parsing, indexing, QA, and page citation tests.
- PostgreSQL dump/restore and Qdrant snapshot/restore were executed. See
  `docs/backup-restore-audit.md`.
- `GET /api/v1/capabilities` exposes available, disabled, and degraded capabilities.

Human review UI: `http://localhost/api/v1/ui/gold-review`. Only explicit reviewer
actions can approve records; rebuilding the dataset always leaves them pending.

面向科研论文的 RAG 与证据化研究助手。当前版本为 `v0.9.0-rc1` 候选：八周功能骨架已完成，第 9 阶段正在固化真实部署、端到端、评测、OCR 和稳定性证据。项目没有宣称达到 v1.0.0。

## Release Candidate 状态

| 能力 | 实际状态 |
|---|---|
| Docker Compose：PostgreSQL、Qdrant、Redis、API、Nginx | 已在本机 Linux daemon 真实构建、启动和重启验证 |
| PDF 上传、SHA-256 去重、PyMuPDF、结构化 JSON/JSONL、页面图片 | 已验证 |
| 结构化 Chunk、BM25、Dense、RRF、元数据过滤、相邻上下文、Trace | 已验证 |
| Embedding | `HashEmbeddingProvider(384)` 确定性基线；不是生产语义模型 |
| Reranker | `LexicalReranker` 确定性基线；本轮消融显示会降低检索质量 |
| 问答 | 抽取式证据拼接与拒答基线；未配置真实生成式 LLM |
| arXiv | 真实 Atom API 已验证 |
| Semantic Scholar | 真实 Graph API；匿名请求本轮返回 429，API Key 未配置 |
| 外部 PDF 下载、导入、解析、索引 | 已真实验证 |
| LangGraph | 真实 `StateGraph`；检查点为 `InMemorySaver`，不是持久化生产实现 |
| Docling / GROBID | 适配器存在；本轮没有安装 Docling 或启动 GROBID 服务 |
| OCR | 主机 Tesseract 5.5.0 对文本/混合/全扫描 PDF 已验证；Docker 镜像内尚未安装，属于 optional fallback |
| 50 条评测集 | 字段和覆盖已统一，但 50/50 均为 `review_status=pending`，0 条人工金标 |
| 30 篇稳定性负载 | 30 篇、100 次检索、100 次问答、3 次 Deep Research、服务重启已运行 |
| Redis | 容器健康与持久卷存在，但应用尚未使用 Redis；API 健康接口也不检查 Redis |

完整限制见 [docs/known-limitations.md](docs/known-limitations.md)。

## 一键启动

要求：Docker Desktop Linux daemon、Docker Compose。

```powershell
Copy-Item .env.example .env
docker compose build --no-cache api
.\scripts\start.ps1
docker compose ps
Invoke-RestMethod http://localhost/api/v1/health
```

- UI：<http://localhost/api/v1/ui>
- OpenAPI：<http://localhost/docs>
- API 健康：<http://localhost/api/v1/health>
- Qdrant：<http://localhost:6333/healthz>

默认 Compose 凭据只用于本地验收，发布到共享环境前必须更换。

## 真实业务闭环

```powershell
$upload = curl.exe -sS -F "file=@paper.pdf;type=application/pdf" `
  http://localhost/api/v1/papers/upload | ConvertFrom-Json
$paperId = $upload.paper.id

Invoke-RestMethod -Method Post "http://localhost/api/v1/papers/$paperId/index"

$retrieve = @{
  query = "What method does this paper propose?"
  filters = @{ paper_ids = @($paperId) }
  recall_k = 20
  top_k = 5
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post http://localhost/api/v1/retrieve `
  -ContentType application/json -Body $retrieve

$qa = @{
  question = "What method does this paper propose?"
  paper_ids = @($paperId)
  top_k = 5
} | ConvertTo-Json
Invoke-RestMethod -Method Post http://localhost/api/v1/qa `
  -ContentType application/json -Body $qa
```

上传会输出 `paper_metadata.json`、`paper_blocks.jsonl`、`parse_manifest.json`、`parse_report.md`、`paper_analysis.json` 和 `page_assets/`。`parse_manifest.json` 记录 `parser_name`、`is_ocr`、`ocr_confidence`、`source_pages` 和 `parse_warnings`；每个 Block 记录 `source_page`、`is_ocr` 和 `ocr_confidence`。

## OCR optional fallback

本机验收使用：

```powershell
$env:TESSERACT_EXE='D:\Program Files\Tesseract-OCR\tesseract.exe'
$env:TESSDATA_PREFIX='D:\Program Files\Tesseract-OCR\tessdata'
$env:PATH='D:\Program Files\Tesseract-OCR;' + $env:PATH
python scripts\run_ocr_audit_v1.py
```

当前 Dockerfile 不安装 Tesseract，因此容器内扫描 PDF 不能作为已验证能力。PyMuPDF OCR API 不返回词级置信度，`ocr_confidence` 会如实记录为 `null` 并附 warning。

## RC 可复现验收

```powershell
python scripts\build_gold_set_v1.py
python scripts\run_evaluation_v1.py
python scripts\run_end_to_end_v1.py
python scripts\run_ocr_audit_v1.py
python scripts\run_stability_v1.py
python -m pytest
python -m ruff check .
```

关键产物：

- [RC 部署与业务验收](docs/release-candidate-audit.md)
- [真实端到端记录](docs/end-to-end-run-v1.md)
- [端到端报告](artifacts/demo-research-report-v1.md)
- [端到端 Trace](artifacts/demo-trace-v1.json)
- [评测报告](docs/evaluation-report-v1.md)
- [评测 JSON](data/evaluation/results-v1.json)
- [评测 CSV](data/evaluation/results-v1.csv)
- [OCR 报告](docs/ocr-audit-v1.md)
- [稳定性报告](docs/stability-report-v1.md)
- [人工复核指南](docs/gold-set-review-guide.md)

## 开发

Python 3.12：

```powershell
uv sync --extra dev
uv run pytest
uv run ruff check .
```

架构与数据流见 [docs/architecture.md](docs/architecture.md)、[docs/pdf-rag-data-flow.md](docs/pdf-rag-data-flow.md) 和 [docs/langgraph-workflow.md](docs/langgraph-workflow.md)。
