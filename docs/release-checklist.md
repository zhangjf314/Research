# Release Checklist

## Portfolio v1.0 evaluation policy update

- [x] `gold-dev-v1`: 50/50 records are human processed and approved. This is an
  internal development evaluation set, not a blind holdout.
- [x] Formal metrics must still count only `review_status=approved` records.
- [x] `retrieval-diagnostic-v1`: 27 claim-level records remain diagnostic and
  are not described as blind.
- [x] Real Production Embedding and the Production collection are available for
  the current retrieval path.
- [x] Real LLM provider preflight passed for SiliconFlow `Qwen/Qwen3-8B`.
- [x] Full QA is no longer blocked solely by the absence of a 50-record strict
  blind shadow holdout.
- [x] `shadow-holdout-pilot-v1` is optional/recommended: 10-15 new samples, not
  a hard Portfolio gate.
- [x] Strong generalization claims remain forbidden without a future independent
  blind benchmark.

Portfolio-safe wording:

> 基于 50 条人工审核的内部评测数据完成检索和问答评测

Forbidden wording:

- 在严格盲测集上证明了泛化能力
- 达到生产级泛化
- 通过大规模独立 benchmark

Current retrieval release state:

- `READY_FOR_FULL_QA=true`
- `RETRIEVAL_GENERALIZATION_EVIDENCE=DIAGNOSTIC_ONLY`
- `STRONG_GENERALIZATION_CLAIM_ALLOWED=false`

## Stage 10 evidence

- [x] Baseline and Production profiles with explicit provider configuration failure.
- [x] Separate logical collections; 34/34-paper Hash rebuild completed with 2,062 points.
- [x] Reranker disabled by default; lexical implementation/history retained.
- [x] Human review workbench and immutable pending-by-script policy; actual 0/50 approved.
- [x] Structured Claim output, evidence binding, support filtering, and latency breakdown.
- [x] Redis TTL cache/rate limiting/import lock/degradation and health usage telemetry.
- [x] PostgreSQL LangGraph checkpoint and API replacement recovery for fixed thread ID.
- [x] Semantic Scholar 429 retry telemetry, Redis hit, and arXiv controlled fallback.
- [x] Capability endpoint and Docker text/mixed/scanned PDF OCR revalidation.
- [x] PostgreSQL dump/restore and Qdrant snapshot/restore.
- [x] 182.7-second bounded soak: 276 queries, 4 imports, 3 research runs, 1 API restart,
  0 failures.
- [ ] 50/50 records approved by human reviewers.
- [ ] Real production Embedding executed and evaluated.
- [ ] Real OpenAI-compatible LLM executed with Token/cost/latency evidence.
- [ ] Cross-Encoder executed and accepted, or formally kept disabled after gold evaluation.
- [ ] Seven-way production evaluation on approved data.
- [ ] Production Deep Research report with real-model usage.
- [ ] Longer production-representative soak and secrets/TLS deployment review.

**Decision: not v1.0.0.**

## v0.9.0-rc1

- [x] Docker Desktop Linux daemon 真实运行。
- [x] `docker compose config` 通过。
- [x] API 镜像无缓存构建通过。
- [x] PostgreSQL、Qdrant、Redis、API、Nginx 真实启动并分别健康。
- [x] Alembic 迁移执行。
- [x] 上传、解析、切分、Embedding、Qdrant、检索、Rerank、问答和引用闭环。
- [x] PostgreSQL 与 Qdrant 重启持久化。
- [x] 错误 PDF、外部超时/限流、Redis 不可用场景记录。
- [x] 固定主题端到端报告与 Trace。
- [x] 50 条评测字段和覆盖统一，全部保持 pending。
- [x] 六组消融真实计算并输出 JSON/CSV/Markdown。
- [x] 主机三类 PDF OCR 验收。
- [x] 30 篇、100 次检索、3 次 Deep Research 稳定性负载。
- [x] 完整测试、Ruff、compileall 和最终镜像重建在最后代码变更后通过。

## v1.0.0 P0（当前未完成）

- [ ] 50 条评测数据完成人工 PDF 复核、批准与必要的 adjudication。
- [ ] 配置并验收真实 LLM，记录模型、Token、费用和生成质量。
- [ ] 配置生产 Embedding；替换或默认关闭已证实降质的词法 Reranker。
- [ ] Docker 容器内 Tesseract OCR 验收通过，或从发布范围明确移除容器 OCR。
- [ ] Semantic Scholar 使用有效 API Key 成功复验。
- [ ] Redis 接入真实用途并纳入健康/降级策略，或从架构声明和 Compose 中移除。
- [ ] LangGraph 使用持久 checkpoint 并验证进程重启恢复。
- [ ] PostgreSQL 备份恢复与 Qdrant snapshot/restore 演练。
- [ ] 更换 Compose 默认口令并完成 secrets/TLS/访问控制检查。
- [ ] 运行长时间 soak，确认内存趋势和失败恢复。

只有 P0 全部完成并有可复现证据后，才能评估是否发布 v1.0.0。
