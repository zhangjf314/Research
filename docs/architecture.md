# 系统架构

```mermaid
flowchart LR
    U["研究人员 / 知识工作者"] --> N["Nginx"]
    N --> API["FastAPI API + MVP UI"]
    API --> ING["论文导入与解析"]
    API --> RAG["Hybrid RAG"]
    API --> AGENT["LangGraph Deep Research"]
    ING --> FS["PDF / JSON / 页面资产"]
    ING --> PG["PostgreSQL 元数据"]
    ING --> QD["Qdrant 向量索引"]
    RAG --> QD
    RAG --> BM25["BM25 Sparse Index"]
    RAG --> TRACE["Retrieval Trace / Usage Event"]
    AGENT --> RAG
    AGENT --> EXT["arXiv / Semantic Scholar"]
    AGENT --> REPORT["带引用研究报告"]
    API -. "尚未接入应用" .-> REDIS["Redis（仅部署与容器健康）"]
```

## 设计原则

- 解析质量、引用可靠性和检索效果优先于界面包装。
- 所有分析字段和研究结论绑定论文、章节、页码和原文证据。
- Dense、Sparse、Fusion、Rerank 和最终上下文均可追踪。
- 外部搜索、自动导入和 Agent 循环受预算与停止条件约束。
- Provider 接口隔离本地基线与生产模型，便于替换 BGE-M3、Cross-Encoder 和 LLM。
