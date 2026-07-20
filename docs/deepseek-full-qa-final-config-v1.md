# DeepSeek Full QA Final Config v1

- Branch/commit: `eval/retrieval-recall-benchmark-v1` / `a1ac66c19d83d4831f950fc58f3b5f764858ff10`
- Dataset/hash: `gold-dev-v1` / `a196fc0c40823dd66b3972cf1d455d647325a20872cfe1f81685b967ec4e2e8d`
- Provider/model: `deepseek` / `deepseek-v4-flash`
- Prompt: `qa-production-v1`
- Retrieval: `{'recall_k': 20, 'top_k': 10, 'context_selector': 'production-api-context-builder', 'context_token_budget': 12000, 'index_version': 'jina-embeddings-v5-text-small-v1', 'collection': 'papers_jina_eval34_v2__20260713152149', 'score_threshold': 0.12}`
- Embedding: `{'provider': 'jina', 'model': 'jina-embeddings-v5-text-small', 'dimension': 1024}`
- Qdrant collection: `papers_jina_eval34_v2__20260713152149`
- Reranker: `disabled`
- QA retry / JSON repair / citation repair: `0` / `false` / `false`
- Transport retry count: `0`
- Config hash: `6c0a713078f4112659d4f4c95a864498335584d5de3e259c2ad6b3b63fc0baac`

Exact-Gold metrics are diagnostic and do not block the Portfolio engineering gate.
This is a 50-item human-reviewed internal development evaluation, not a blind holdout.
