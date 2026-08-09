# Changelog

## [1.1.0-portfolio] - 2026-08-09

### Added

- Research Agent v1 runtime with explicit state, evidence state, dynamic tool
  selection, verification-before-finish, checkpoint/resume, retry bounds,
  budget limits, stop conditions, and trace.
- Frozen 60-unit Workflow vs Agent paired benchmark and Stage 4C validity audit.
- Portfolio final facts, project summary, interview notes, release status, and
  v1.1.0 release notes.

### Changed

- Package/runtime version is `1.1.0+portfolio`; display/tag version is
  `1.1.0-portfolio`.
- README public positioning now emphasizes real engineering closure and
  auditable evidence boundaries without the removed personal job-seeking
  sentence.

### Validation

- Stage 4C.1 final validity audit completed with no new provider, Workflow, or
  Agent runs.
- Structured proxy metrics are valid for portfolio claims.
- Semantic content-level rubric validation remains explicitly incomplete.

## [1.0.1-portfolio] - 2026-07-25

### Fixed

- Render Deep Research reports as sanitized Markdown in the repository UI.
- Replace legacy evidence-dump reporting with structured DeepSeek synthesis.
- Deduplicate evidence and references across report sections.
- Enforce global and section-scoped citation allowlists.
- Add replayable raw-response observability and failed-request usage accounting.
- Add report duplication and cross-section similarity quality gates.

### Validation

- Production Deep Research hotfix live smoke completed successfully.
- Raw response replay, schema validation, citation allowlist validation, and report
  quality gates passed.
- Release tests passed with 647 tests and 1 warning before tagging.

## [Unreleased] - Stage 10

### Added

- Baseline/Production provider profiles, OpenAI-compatible Embedding/LLM adapters, and a
  Cross-Encoder reranker adapter with explicit configuration errors.
- Versioned index registry, staged full rebuild/rollback/switch APIs, and separate Hash
  and Production collection names.
- Claim-level structured QA and retrieval/model latency fields.
- Human gold review workbench, progress report, and quality audit.
- Redis TTL cache, rate limiting, import locking, usage telemetry, and graceful fallback.
- PostgreSQL LangGraph checkpoints with pause/resume by thread ID.
- Capability endpoint, Docker Tesseract, backup/restore audit, and bounded soak runner.

### Changed

- Lexical reranking is disabled by default; Structural + Hybrid is the baseline default.
- Production evaluation refuses to run on pending annotations or missing model providers.

### Not completed

- Real-model execution and 50/50 human gold review remain externally blocked; no v1.0.0
  claim or production quality metric is made.

## [0.9.0-rc1] - 2026-07-13

### Added

- Release Candidate Docker、业务闭环、持久化和失败场景审计。
- 固定主题真实外部搜索、下载、解析、索引、检索、LangGraph 报告和引用 Trace。
- 50 条统一字段的人工复核队列与复核指南；所有条目保持 pending。
- 六组检索/回答消融的 JSON、CSV 和 Markdown 结果。
- 文本、混合、全扫描 PDF 的主机 Tesseract OCR 验收。
- 30 篇、100 次检索、100 次问答、3 次 Deep Research 稳定性负载。
- OCR 来源字段：`parser_name`、`is_ocr`、`ocr_confidence`、`source_page`、`parse_warnings`。

### Changed

- 包版本调整为 `0.9.0-rc1` 候选。
- README 改为基于真实验收的功能状态表。
- UI 和问答固定文案清理为稳定 UTF-8/ASCII 展示。
- Hybrid 延迟指标改为包含 Dense、Sparse、融合、Rerank/上下文的可比口径。

### Known limitations

- 无真实 LLM、生产 Embedding 或 Cross-Encoder。
- 0/50 人工 gold；Semantic Scholar 429；容器 OCR、Redis 应用接入和持久 LangGraph checkpoint 未完成。
