# End-to-End Run v1

- Run ID: `e2e-v1-20260713T094429Z`
- Fixed topic: retrieval augmented generation for scientific literature review: methods evaluation and limitations
- Started: `2026-07-13T09:44:29.548205+00:00`
- Completed: `2026-07-13T09:44:56.498028+00:00`
- Papers imported/indexed: 3
- Parse success rate: 100.0%
- Indexed chunks: 126
- Retrieval rounds: 3
- LangGraph local-search iterations: 1
- Tool calls: 11
- LLM tokens / cost: 0 / $0.00 (no real LLM is configured)
- Elapsed: 26.95 s
- Citation validation pass rate: 100.0%

## Provider truth table

- query_rewriter: deterministic QueryRewriter
- external_search: ['arXiv Atom API', 'Semantic Scholar Graph API']
- pdf_download: httpx via CachedRetryClient
- parser: ParserRouter (PyMuPDF baseline; OCR fallback when routed)
- embedding: HashEmbeddingProvider(dimensions=384)
- vector_store: Qdrant HTTP
- sparse: BM25Retriever
- fusion: RRF(k=60)
- reranker: LexicalReranker
- report: deterministic evidence-template generator (no LLM)
- workflow: LangGraph with InMemorySaver

## External source errors

- semantic_scholar: `HTTPStatusError: Client error '429 ' for url 'https://api.semanticscholar.org/graph/v1/paper/search?query=retrieval+augmented+generation+for+scientific+literature+review+methods&limit=8&fields=title%2Cabstract%2Cauthors%2Cyear%2Cvenue%2CexternalIds%2Curl%2CcitationCount%2CopenAccessPdf%2CpublicationDate'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/429`

## Acceptance boundary

The download, PDF parsing, indexing, Qdrant storage, hybrid retrieval, reranking, LangGraph execution, report file, and citation-marker validation are real. Query rewriting, embedding, reranking, and report generation are deterministic baselines. This run therefore validates system wiring, not production-model answer quality.
