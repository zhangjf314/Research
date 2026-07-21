# 三个演示案例

## 案例一：单篇论文阅读

论文：Attention Is All You Need（arXiv:1706.03762）。

1. 上传 PDF，观察 `UPLOADED → PARSING → PARSED`。
2. 打开论文详情页，检查研究问题、方法和实验字段的页码证据。
3. 建立索引并提问 “How does self-attention replace recurrence?”。
4. 展示原文片段、章节、页码和 PDF 定位入口。

验收重点：答案不能脱离检索证据；错误问题应触发拒答。

## 案例二：多论文检索消融

语料：项目内 10 篇真实 arXiv 论文，913 个结构化 Chunk。

1. 打开评测中心。
2. 对比固定 Dense、结构 Dense、BM25、Hybrid 和 Hybrid + Rerank。
3. 展示结构 Dense Hit@1 0.80，相比固定 Dense 0.30。
4. 解释为何当前词法 Rerank 会降低质量，以及如何用 Cross-Encoder 替换。

验收重点：所有简历指标能定位到可重复运行的 JSON/Markdown 报告。

## 案例三：深度研究工作流

主题：long-context language models: methods, results, and limitations。

1. 打开深度研究页面并提交主题。
2. 展示 4 个子问题和研究计划。
3. 展示本地证据覆盖、预算、节点轨迹和引用校验。
4. 打开最终研究报告，抽查引用页码。

验收重点：证据不足时进入外部搜索；预算触顶时停止；报告主要结论均有引用。
# Demo Cases

## Stage 13.39 safe demo scope

Use curated summaries and sanitized API views only. The demo may show health,
capabilities, retrieval traces, final QA summary metrics, and the successful
bounded q003 Deep Research result. It must not claim strong generalization or
show local-only raw traces/provider responses.
