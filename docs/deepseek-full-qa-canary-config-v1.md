# DeepSeek Full QA Canary Config v1

- Provider/model: `deepseek` / `deepseek-v4-flash`
- Sample IDs: `q014, q020, q029, q031, q032, q035, q036, q037, q044, q001, q008, q015, q024, q049, q005`
- Retrieval config: `{'recall_k': 20, 'top_k': 10, 'context_token_budget': 12000, 'index_version': 'jina-embeddings-v5-text-small-v1', 'collection': 'papers_jina_eval34_v2__20260713152149'}`
- Reranker enabled: `False`
- Thinking: `disabled`
- Response format: `json_object`
- Configuration SHA-256: `deb373b2e1ee5fbc3df5032beec1383047f7055b8fed4b551d68cd47ea5b3693`

This canary reuses the same internal development samples as Qwen v2; it is not a blind benchmark.
