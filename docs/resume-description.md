# 简历项目描述

## 精简版

**PaperResearch Agent｜论文 RAG 与深度研究助手**

- 基于 FastAPI、PostgreSQL、Qdrant 和 LangGraph 构建论文研究平台，支持 PDF 结构化
  解析、混合检索、页码引用问答、外部论文搜索和自动化深度研究报告。
- 设计结构优先、长度兜底的 Chunk Pipeline，保留章节、页码、块类型及相邻上下文；
  在 10 篇真实论文、50 条银标候选上，结构化 Dense Hit@1 达 0.80，固定切分为 0.30。
- 实现 Dense + BM25 并行召回、RRF 融合、元数据过滤和 Retrieval Trace；Hybrid
  Recall@10 达 0.90，并通过消融识别词法 Rerank 的负收益。
- 使用 LangGraph 实现证据缺口驱动的研究工作流，加入多维预算和停止条件；真实工作流
  审计覆盖 4 个子问题、20 条证据，引用校验 20/20。

## 指标使用声明

上述指标来自 `silver` 评测候选，仅适合描述工程基线。完成领域专家人工复核前，不应写成
“人工评测准确率”或对外宣称生产效果。
# Portfolio-safe resume wording

**PaperResearch Agent：论文 RAG 与证据化研究助手**

- 构建基于 FastAPI、PostgreSQL、Qdrant、Redis 和 LangGraph 的论文 RAG
  系统，覆盖 PDF 解析、结构化 Chunk、Hybrid Retrieval、证据绑定 QA、
  citation validation 和可追溯评测报告。
- 基于 50 条人工审核的内部评测数据完成检索和问答评测；另使用 27 条
  claim-level diagnostic 数据进行失败分析和检索回归检查。
- 接入真实 Jina Embedding、SiliconFlow `Qwen/Qwen3-8B` provider preflight
  和可切换 Reranker 消融；默认关闭未通过质量/延迟门槛的 Reranker。

Do not claim:

- 严格盲测集证明泛化能力。
- 生产级泛化。
- 大规模独立 benchmark 通过。
