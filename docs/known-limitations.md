# Known Limitations

## Stage 13.39 portfolio blockers

- The project has not been tagged or released as `v1.0.0-portfolio`.
- DeepSeek Full QA and a bounded q003 Deep Research smoke passed, but
  PostgreSQL production checkpoint recovery v2 has not been executed.
- PostgreSQL backup/restore v2 has not been executed against the current
  `0.9.0rc3` runtime.
- Qdrant snapshot restore v2 and fixed-query Top-K comparison have not been
  executed.
- Docker reports Tesseract/OCR capability as available, but the Stage 13.39
  Docker text/mixed/scanned end-to-end OCR roundtrip has not been executed.
- The required Portfolio 30-minute stability test has not been executed.
- The stability test window is intentionally bounded for a personal portfolio.
  Passing it may only support this statement: "Within this 30-minute test
  window, no obvious sustained abnormal memory growth was observed." It must
  not be described as proof of long-term stability or a production-grade
  endurance test.
- Broad git-history secret matching still requires manual line-level review
  before any public release.
- `SEMANTIC_CLAIM_SUPPORT_AUDIT=NOT_FORMALLY_VALIDATED` and
  `STRONG_GROUNDING_CLAIM_ALLOWED=false` remain in force.

## Portfolio evaluation limitations

- The main evaluation set, `gold-dev-v1`, contains 50 human-approved records and
  is an internal development evaluation set. It is not a blind holdout, public
  benchmark, or strict generalization benchmark.
- `retrieval-diagnostic-v1` contains 27 claim-level records used for diagnostic
  failure analysis and regression checks. It has been inspected during
  development and must not be described as blind.
- `shadow-holdout-pilot-v1` has not been created. It is recommended as a
  10-15-sample small blind pilot, but it is not required for Portfolio Full QA.
- `RETRIEVAL_GENERALIZATION_EVIDENCE=DIAGNOSTIC_ONLY`; strong generalization
  claims are not allowed.
- README, demo, and resume wording must disclose that there is no large-scale
  independent blind benchmark result.

## Stage 10 current blockers (2026-07-13)

- Human gold remains **0/50 approved**. All pending records are excluded from formal
  production statistics.
- No real model credentials are configured. Production Embedding, Cross-Encoder, and
  OpenAI-compatible LLM have adapters but no executed Token, cost, latency, or quality
  evidence on this machine.
- The lexical reranker remains disabled by default because the RC ablation reduced
  retrieval quality. A cross-encoder must beat or justify its trade-off on approved data
  before it can become the production default.
- Semantic Scholar anonymous access returned HTTP 429. Retry/backoff, Redis cache, and
  arXiv fallback were verified; API-key success remains unverified.
- Docling is not installed and GROBID is not configured. The capability endpoint reports
  these as unavailable/degraded; PyMuPDF is the verified primary parser.
- The completed 182.7-second soak had no failures and survived an API restart, but is a
  bounded local run, not evidence against slow leaks over production-scale durations.
- Compose still contains local development database credentials and has no TLS/auth
  boundary; it must not be exposed directly to an untrusted network.

## P0：v1.0.0 前必须处理

- 50 条评测集全部 `pending`，0 条人工批准。正式检索、回答和引用质量指标尚不存在。
- 没有配置真实生成式 LLM；报告与问答是确定性证据模板/抽取式基线。
- Hash Embedding 不是生产语义向量模型；Lexical Reranker 的本轮消融显著降低 Hit/MRR/NDCG。
- Docker 镜像未安装 Tesseract；扫描 PDF 在部署容器中的 OCR fallback 未验证。
- Semantic Scholar 匿名访问本轮返回 429，需要 API Key、限流预算和成功复验。
- Redis 尚未被应用使用，也未纳入 API 健康检查。Redis 故障不会触发 API degraded。
- LangGraph 使用 `InMemorySaver`，任务状态无法跨 API 进程重启恢复。

## P1：生产化前需要处理

- Docling 未安装，GROBID 服务未启动；仅适配器和结构测试存在。
- OCR 置信度为 `null`：当前 PyMuPDF OCR API 不暴露词级 confidence。
- 混合 PDF 触发时会对整篇执行 OCR，而不是只 OCR 低文本页面。
- PDF 解析的多栏阅读顺序、复杂表格、公式和参考文献结构仍依赖基础启发式。
- 抽取式回答可能把不理想的 Top-1 Chunk 直接作为正文；引用存在不等于答案充分。
- Qdrant 小集合使用 full scan，稳定性数据不代表大规模 HNSW 性能。
- 只验证了 Docker volume 重启持久化；未演练 PostgreSQL 备份恢复或 Qdrant snapshot/restore。
- 稳定性测试约 168 秒，不是长时间 soak；离散内存快照无法排除缓慢泄漏。
- Compose 包含本地默认数据库口令，不能直接用于共享或公网环境。
- 没有认证、授权、配额、TLS、恶意文件扫描或多租户隔离。

## 指标解释限制

- `gold-set-v1` 文件当前实质是人工复核队列，不是已批准金标。
- Faithfulness 等指标由词项重叠启发式计算，不是人工裁决或模型裁判。
- 评测延迟是本地进程内基线，不含真实模型网络延迟。
- Token 和费用为 0 的原因是没有调用计费 LLM，而不是成本优化结论。
