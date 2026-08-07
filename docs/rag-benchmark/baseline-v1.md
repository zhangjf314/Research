# RAG Benchmark Baseline v1

- git_commit: `f97746e84b98d6b4e07984a3abbdab206f156839`
- runtime_version: `1.0.1+portfolio`
- baseline_config_hash: `8817891ed73fc0c0fa2f3a7fc90baf3591b3ab0006934bbb058cbff87e657c94`
- LLM: `deepseek/deepseek-v4-flash`
- Embedding: `jina/jina-embeddings-v5-text-small`
- Retrieval collection: `papers_production_v1`
- top_k: `5`
- chunking: max `400`, overlap `60`
- reranker_enabled: `False`
- query_rewrite: `disabled`
- context_selection: current production context builder; no Stage 1 changes

Stage 1 freezes this configuration. Do not change these parameters to improve benchmark scores.
