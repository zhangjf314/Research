# 实施进度

## 第 0 阶段：项目初始化

- [x] Python 3.12 项目与基础目录
- [x] FastAPI 配置与版本化路由
- [x] PostgreSQL 论文模型
- [x] Qdrant/PostgreSQL 健康检查
- [x] Docker Compose
- [x] Alembic 初始迁移
- [x] README、开发规范与测试骨架
- [ ] 收集 10～20 篇不同类型且许可允许的测试论文
- [ ] 在真实 Docker 环境执行端到端连接验收

## 下一步：第 1 周 PDF 解析流水线

- [x] PDF 上传与类型/内容/大小校验
- [x] SHA-256 文件哈希与重复论文识别
- [x] 统一 `PaperParser` 接口与 `ParserRouter`
- [x] PyMuPDF 基线解析器
- [x] 标题、章节、正文块、页码与边界框归一化
- [x] `paper_metadata.json`、`paper_blocks.jsonl`、`parse_report.md`
- [x] 低文本页 OCR 回退提示
- [x] Docling 解析适配器（可选 `parsing` extra）
- [x] GROBID TEI 元数据、正文与参考文献适配器
- [x] OCR 自动回退代码路径与低文本页触发策略
- [x] 页面 PNG 资产生成
- [x] 10 篇真实 arXiv 论文解析审计（10/10 成功）
- [ ] 部署 Tesseract 运行时并完成扫描论文 OCR 验收
- [ ] 启动 GROBID 服务并完成真实请求验收

## 第 2 周：结构化切分与基础 RAG

- [x] 结构优先、长度兜底的 Chunk 策略
- [x] 固定 Token 切分对照组
- [x] Chunk 保留论文、章节、页码、类型和块关联
- [x] 父级章节与前后相邻上下文补全
- [x] Embedding Provider 抽象与本地确定性基线
- [x] Qdrant Dense 向量写入、论文过滤和 Top-K 检索
- [x] 基础问答接口与 PDF 页码定位引用
- [x] 无足够证据时拒答
- [x] 10 篇真实论文切分审计
- [ ] 接入 BGE-M3 生产级 Embedding
- [ ] 在真实 Qdrant 服务执行全量索引验收

## 第 3 周：混合检索与 Rerank

- [x] BM25 Sparse Retrieval
- [x] Dense 与 Sparse 并行召回
- [x] RRF 融合及双路名次记录
- [x] 可替换 Reranker 抽象与本地词法基线
- [x] 论文、章节、块类型和页码元数据过滤
- [x] 前后相邻上下文补全与上下文长度预算
- [x] Dense、Sparse、Fusion、Rerank、Final Context 完整 Trace
- [x] Trace JSONL 持久化与混合检索 API
- [x] 10 条真实论文检索冒烟评测
- [ ] 接入 BGE Cross-Encoder Reranker
- [ ] 建立人工标注的正式检索评测集

## 第 4 周：论文分析与 MVP

- [x] 研究背景与研究问题提取
- [x] 主要贡献与方法提取
- [x] 实验设置、结果和局限性提取
- [x] 数据集、基线、指标、超参数、硬件和消融信息卡片
- [x] 所有生成字段绑定原文块、章节和页码证据
- [x] 解析完成后自动生成 `paper_analysis.json`
- [x] 论文详情与分析 API
- [x] 论文详情 HTML 页面
- [x] 基础评测中心 HTML 页面
- [x] 10 篇真实论文分析审计
- [ ] 接入生成模型进行忠实的跨证据综合摘要
- [ ] 完成多篇论文问答与比较页面

## 第 5 周：论文搜索与自动导入

- [x] arXiv Atom API 搜索客户端
- [x] Semantic Scholar Graph API 搜索客户端
- [x] 查询规范化与多查询改写
- [x] 多来源并行搜索与统一候选模型
- [x] DOI、arXiv ID 和规范化标题交叉去重
- [x] 标题、摘要、时间、引用量与开放 PDF 综合评分
- [x] 年份和开放获取筛选
- [x] TTL 搜索缓存
- [x] 429、5xx、网络错误和超时重试
- [x] PDF 一键下载、解析、分析与去重导入
- [x] 真实双源搜索审计
- [ ] Redis 分布式缓存和后台任务队列
- [ ] 下载任务状态页面和人工重试按钮

## 第 6 周：LangGraph 深度研究工作流

- [x] 完整 `ResearchState` 与显式预算模型
- [x] 研究问题理解、子问题拆分和研究计划
- [x] 本地论文检索节点
- [x] 外部论文搜索与候选筛选节点
- [x] 可选自动下载、导入、解析后重新检索
- [x] 证据覆盖和缺口判断
- [x] 多论文证据综合与潜在冲突识别
- [x] 带页码引用的多章节研究报告
- [x] 引用一致性校验
- [x] 最大迭代、搜索、论文、证据、Token 和无新增证据停止条件
- [x] LangGraph `InMemorySaver` 线程级检查点
- [x] 深度研究 API 与状态/报告持久化
- [x] 10 篇真实论文工作流审计
- [ ] 接入生产 LLM 进行问题拆分和跨论文综合
- [ ] 使用 PostgreSQL checkpointer 实现跨进程恢复

## 第 7 周：评测、可观测性与消融实验

- [x] 50 条证据绑定银标评测候选集
- [x] 人工标注规范、复核状态与发布门槛
- [x] Hit@K、Recall@K、MRR、NDCG@10 和 Block Hit@10
- [x] Answer Relevancy、Faithfulness、Context Precision/Recall
- [x] Citation Coverage、Citation Correctness、Unsupported Claim Rate
- [x] Agent 子问题覆盖、搜索轮数、工具调用、停止与引用指标
- [x] 延迟、Token、估算费用 Usage Event
- [x] 固定切分与结构切分消融
- [x] Dense、Sparse、Hybrid、Hybrid + Rerank 消融
- [x] Markdown 与 JSON 综合评测报告
- [ ] 由领域标注者将 50 条银标全部复核为 `human_reviewed`
- [ ] 接入真实付费模型后记录实际 Token 和费用

## 第 8 周：部署与求职包装

- [x] 非 root API Dockerfile 和健康检查
- [x] PostgreSQL、Qdrant、Redis、API、Nginx Docker Compose
- [x] PowerShell 一键启动、停止和环境检查
- [x] GitHub Actions 测试、静态检查和 Compose 校验
- [x] 统一异常响应、请求 ID 与安全错误信息
- [x] MVP 控制台、论文库、搜索、研究和评测页面
- [x] 系统架构图、PDF/RAG 数据流图和 LangGraph 工作流图
- [x] 三个演示案例和三分钟录屏脚本
- [x] 部署手册、发布清单、面试讲解和简历描述
- [ ] 启动 Docker Desktop 后完成本机镜像构建与服务健康验收
- [ ] 实际录制三分钟演示视频
