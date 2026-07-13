# Release Candidate Audit

验收日期：2026-07-13（Asia/Shanghai）
候选版本：`v0.9.0-rc1`
结论：满足“可发布 RC 供进一步人工验收”的条件；不满足 v1.0.0 条件。

## 1. 实际执行命令与结果

### Docker 与部署

```powershell
docker version
docker compose config
docker compose build --no-cache api
.\scripts\start.ps1
docker compose ps
docker compose exec -T postgres pg_isready -U paper -d paper_research
docker compose exec -T redis redis-cli ping
Invoke-RestMethod http://localhost:6333/healthz
Invoke-RestMethod http://localhost/api/v1/health
Invoke-WebRequest http://localhost/ -UseBasicParsing
```

- 首次 `docker version`：客户端 29.5.3 可用，daemon 管道不存在。
- 启动 Docker Desktop 后：Docker Desktop 4.78.0、Linux Engine 29.5.3 正常。
- `docker compose config`：退出码 0，五个服务和四个持久卷解析成功。
- 无缓存 API 构建：退出码 0，约 150.5 秒，生成 `research-api:latest`。拉取基础层时发生一次 EOF 并自动恢复。
- 最终代码再次无缓存构建：退出码 0，117 秒；镜像内 wheel 为 `paper-research-agent-0.9.0rc1`。
- 一键启动：退出码 0；Alembic 执行 `0001 create papers table`。
- PostgreSQL：`accepting connections`；Redis：`PONG`；Qdrant：`healthz check passed`；Nginx：HTTP 200。
- API：`healthy`，PostgreSQL/Qdrant 均为 `up`。API 不检查 Redis，见限制。

### 真实业务闭环

使用公开论文 `data/raw/audit/1706.03762.pdf`：

```powershell
curl.exe -F "file=@data/raw/audit/1706.03762.pdf;type=application/pdf" `
  http://localhost/api/v1/papers/upload
Invoke-RestMethod -Method Post http://localhost/api/v1/papers/2537b3aa-d6aa-4a4a-aac1-477fc58bc3d9/index
Invoke-RestMethod -Method Post http://localhost/api/v1/retrieve -ContentType application/json -Body $retrieve
Invoke-RestMethod -Method Post http://localhost/api/v1/qa -ContentType application/json -Body $qa
```

- 论文 ID：`2537b3aa-d6aa-4a4a-aac1-477fc58bc3d9`。
- 上传、哈希去重、15 页 PyMuPDF 解析、结构化 JSONL、分析和页面 PNG：通过。
- 结构化切分：25 Chunk。
- Hash Embedding 384 维、Qdrant upsert：通过；集合 points 从 0 增至 25。
- Dense+BM25、RRF、Lexical Rerank、相邻上下文和 Retrieval Trace：通过。
- 抽取式问答返回 5 条引用，含 paper ID、section、page range、quote、score 和 PDF page anchor。
- 质量边界：该闭环使用确定性 Hash Embedding、词法 Reranker 和抽取式回答，不是生产模型质量验收。

第一次调用误用了不存在的 `/api/v1/retrieval/retrieve` 与 `/api/v1/retrieval/qa`，得到 404；根据 OpenAPI 改为现有 `/api/v1/retrieve` 与 `/api/v1/qa` 后通过。README 已按真实路由更新。

最终镜像又上传了新 PDF `text-native.pdf`：ID `0228c3e3-8630-4f5c-b8dc-b83c13eabe5a`，`parse_manifest.json` 包含预期 OCR 来源字段，索引 1 个 Chunk，Trace `8284d045-7233-4900-9cb1-cce1d5f5f1b6`。宽泛问题因低于阈值正确拒答；使用与文档证据完全匹配的问题时 score 0.845154，返回 1 条第 1 页引用。重复上传同一文件返回 `duplicate=true` 和相同 paper ID。

### 持久化

```powershell
docker compose restart postgres qdrant api nginx
```

- 重启前后 PostgreSQL 中同一论文均为 `READY/READY`。
- 重启前后 Qdrant `paper_chunks` 均为 green，`points_count=25`。
- 30 篇稳定性负载后再次重启五个服务，健康检查恢复。

### 失败场景

```powershell
curl.exe -i -F "file=@data/fixtures/invalid.pdf;type=application/pdf" `
  http://localhost/api/v1/papers/upload
docker compose stop redis
Invoke-RestMethod http://localhost/api/v1/health
docker compose start redis
```

- 错误 PDF：HTTP 422，错误码 `HTTP_422`，消息 `file content is not a PDF`，包含 request ID。
- 外部超时：`CachedRetryClient` 对不可路由地址使用 0.2 秒超时、2 次尝试，实际 672.193 ms 后返回 `ReadTimeout`。
- Semantic Scholar：真实匿名调用在重试后返回 HTTP 429，arXiv 同次调用仍返回 8 个候选。
- Redis 停止：Redis 容器确实停止，但 API 仍返回 healthy；原因是应用尚未使用/检查 Redis。恢复后 `PONG`。此项是限制，不是通过。

## 2. Provider 真值审计

| 能力 | 当前真实实现 | 结论 |
|---|---|---|
| LLM | 未配置；分析/报告/问答为规则或证据模板 | 占位基线 |
| Embedding | `HashEmbeddingProvider(384)` | 真实运行的确定性 fallback，不是语义模型 |
| Reranker | `LexicalReranker` | 真实运行的词法 fallback；消融中降质 |
| arXiv | `export.arxiv.org/api/query` | 真实调用通过 |
| Semantic Scholar | Graph API | 真实调用，匿名限流 429；Key 未配置 |
| PDF 下载 | `httpx` + retry/cache | 真实调用通过 |
| PDF 解析 | PyMuPDF；Tesseract optional fallback | 主链路通过；容器 OCR 未通过 |
| Qdrant | Docker Qdrant 1.12.5 HTTP | 真实入库、检索、持久化通过 |
| PostgreSQL | Docker PostgreSQL 16 | 真实迁移、写入、重启持久化通过 |
| Redis | Docker Redis 7 | 容器真实健康；应用未使用 |
| LangGraph | `StateGraph` + `InMemorySaver` | 图执行真实；检查点不持久 |
| Docling | 可选适配器 | 本轮未安装/未运行 |
| GROBID | HTTP 适配器 | 本轮未启动服务 |
| 测试 Fixture/Mock | 单元测试中的 HTTP transport、PDF fixture、内存 Qdrant | 仅测试使用，不作为真实验收证据 |

## 3. 端到端与负载证据

- 固定主题端到端：3 篇下载/解析/索引，126 Chunk，3 轮检索，11 次工具调用，LangGraph 1 次本地检索迭代，引用检查 100%，26.95 秒，LLM Token/费用为 0。
- 稳定性：30 篇唯一论文，解析成功率 100%，100 次检索、100 次问答、3 次 Deep Research，零失败，服务重启恢复。
- OCR：主机文本型、混合型、全扫描三类 PDF 均真实运行；Docker 内未安装 Tesseract。
- 评测：六组结果由 `scripts/run_evaluation_v1.py` 计算；50 条均 pending，指标只作为 provisional baseline。

## 4. 通过、失败与阻塞

### 已通过

- Docker 无缓存构建、五服务启动、数据库迁移和五服务实际健康检查。
- 上传到引用定位真实业务闭环。
- PostgreSQL/Qdrant 容器重启持久化。
- 错误 PDF 统一错误响应与外部请求超时/限流记录。
- 外部 arXiv 搜索、PDF 下载、导入、解析、索引和固定主题研究链路。
- 六组可复现消融、30 篇中等负载、100 次检索、3 次 Deep Research。
- 主机 OCR 三类 PDF 链路。

### 失败或阻塞

- Redis 不可用时 API 不降级，说明它未接入应用健康和业务。
- Semantic Scholar 匿名访问本轮受 429 限流，未配置 API Key。
- 真实 LLM、生产 Embedding、Cross-Encoder Reranker 均未配置。
- 50 条评测数据 0 条人工批准，不能作为正式 gold 指标。
- Docker 镜像内 OCR 未安装；Docling/GROBID 未实际运行。
- LangGraph 使用内存检查点；没有跨进程恢复。
- 内存只采集离散快照，没有长时间 soak，不能证明无持续增长。
- 未执行 Qdrant snapshot/restore 演练，只验证了 volume 重启持久化。

## 5. RC 判断

当前满足 `v0.9.0-rc1`：核心演示链路、容器部署和可复现工程指标已有真实证据，同时所有 fallback 和阻塞均已标记。当前不满足 v1.0.0；详见 `docs/known-limitations.md` 与 `docs/release-checklist.md`。
