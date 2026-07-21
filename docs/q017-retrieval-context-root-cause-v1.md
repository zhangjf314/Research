# q017 Retrieval Context Root Cause v1

- Gold block `b000033` is an Abstract block on page 1 and is reasonable.
- Before Stage 13.32 it was recalled only at deeper ranks (dense 43, sparse 29, fusion 40) and did not enter final context.
- Root cause: contribution intent used generic similarity plus shallow context candidate cutoff; no q017-specific hardcoding was used.
- After the generic contribution route, q017 gold in context: `True`.
- Top-3 before the fix did not contain equivalent complete evidence for all required claims.
- Classification: `RETRIEVAL_TRUE_MISS`, `CONTEXT_TOP_K_TOO_SMALL`, `QUERY_ROUTING_MISS`.
