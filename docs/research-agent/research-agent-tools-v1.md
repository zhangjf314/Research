# Research Agent Tools v1

Initial tool registry:

- retrieve_evidence
- inspect_evidence
- inspect_paper
- verify_evidence
- finish

Retrieval wraps the existing frozen RAG service boundary. Reranker, query
rewrite and query decomposition remain disabled.

Comparability: `Agent v1 changes action timing and state-conditioned tool selection only; it does not add a new retriever, reranker, embedding model, query rewrite, query decomposition module, or information source.`
