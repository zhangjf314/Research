# RAG Quality Optimization Portfolio Summary

The closed RAG Quality v3-v5 program establishes a clear production result:
the retrieval foundation is sound, but the evaluated post-retrieval policies are
not robust enough to deploy.

Gold-free runtime and evaluation-only attribution were validated. Candidate
generation remained strong. Pointwise reranking and listwise selection improved
average development measurements, but neither supplied safe promotion evidence:
the reranker failed fresh-blind generalization and the listwise policy failed
tail safety. Facet-aware variable-K selection sharply reduced non-Gold context
and improved precision, but omitted required evidence. A local verifier did not
reliably estimate calibrated evidence sufficiency.

The deliberate outcome is `NO_CANDIDATE_PROMOTED` and
`PRODUCTION_P0_RETAINED`. P0 remains dense retrieval plus BM25 and RRF with
rank-order packing. Experimental rerankers, selectors, and verifiers are not
in the production default. FINAL_BLIND_V4, FINAL_BLIND_V5, and Full QA were not
consumed or run.
