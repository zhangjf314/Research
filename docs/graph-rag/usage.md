# Using the optional runtime GraphRAG

Install and configure the existing application as in the [README](../../README.md).
Use your own authorized PDFs, PostgreSQL and a compatible Qdrant server/client.
No graph database, training, paid graph builder or extra model dependency is needed.

## Build from ACTIVE sources

Keep Hybrid as the default. Add the following to your local configuration only
when enabling this feature:

```dotenv
GRAPH_RAG_ENABLED=true
GRAPH_RAG_ALLOW_HYBRID_FALLBACK=false
```

First admit the full intended batch through the existing D6 operator workflow,
in a fresh DATA_DIR/collection. Upload intake alone is not approval. Then, using
the same environment, build the structural graph/Wiki:

```powershell
python -m paper_research.admission status
python -m paper_research.evidence_graph.runtime build
# Compose users can run the same command in the existing api container:
docker compose exec api python -m paper_research.evidence_graph.runtime build
```

The builder writes `DATA_DIR/graph-runtime/index.json` atomically under the D6
lease. It records zero semantic relations/model requests. Keep it local and out
of Git. Rebuild explicitly after a legitimately admitted new generation or
index identity change; stale graphs are refused, never silently rebound. Do not
edit the graph file to add unverified facts.

## API requests

Use the paper UUIDs returned by your admission. The JSON examples below use
placeholders; do not copy them as real identities.

```json
POST /api/v1/retrieve
{
  "query": "Relate the method definition to its experimental configuration.",
  "filters": {"paper_ids": ["<active-paper-uuid>"]},
  "top_k": 5,
  "graph_rag": {"max_hops": 2, "max_nodes": 256, "max_edges": 256,
                "max_context_chars": 16000, "fallback_to_hybrid": false}
}
```

Omit `graph_rag` to keep the original Hybrid response/behavior. `{}` opts in with
defaults. The opt-in response adds `graph_rag` and the existing canonical evidence
contract. `USED` means validated structural traversal occurred, not that it
improved retrieval or proved a claim. `NO_GRAPH_PATH` records empty seed coverage.
Inspect `added_chunk_ids`, `lost_hybrid_chunk_ids`, paths and context truncation.
GraphRAG currently supports paper filters only: page/section/block-type filters
are rejected explicitly rather than ignored during expansion.

```json
POST /api/v1/graph/query
{
  "paper_ids": ["<active-paper-uuid>"],
  "seed_chunk_ids": ["<canonical-chunk-uuid>"],
  "options": {"max_hops": 2, "max_nodes": 256, "max_edges": 256}
}
```

This returns bounded paths **and actual canonical evidence** (PDF SHA, chunk,
blocks, physical pages and quote offsets). The first evidence prefix has offsets
`0..len(evidence)` in its canonical chunk, not in a guessed PDF text string.

```json
POST /api/v1/wiki/query
{
  "paper_ids": ["<active-paper-uuid>"],
  "paper_id": "<active-paper-uuid>",
  "max_evidence_blocks": 4,
  "max_chars": 16000
}
```

Wiki bundles include the structural status, original source quotes and provenance
paths. Budget overflow is explicit; a Wiki page is not an approved research claim.

## ResearchAgent

```json
POST /api/v1/research/agent
{
  "query": "Explain how the admitted method and experiment are related.",
  "paper_ids": ["<active-paper-uuid>"],
  "graph_rag": {},
  "generate_draft": false,
  "budget": {"max_steps": 8, "max_tool_calls": 6, "max_provider_requests": 0}
}
```

The existing Runner/Planner calls retrieve_evidence. Its canonical items retain
the original provenance contract; graph receipts appear separately in
`active_provenance.graph_rag.requests` with tool-call/trace IDs. Checkpoints retain
options; resume uses them, and changing/removing the provider's GraphRAG mode is
rejected. The existing repository-relative backend lock must remain available
when launching the API from the project root. This opt-in currently supports the
**text ACTIVE Agent path**, not the readonly observation or native multimodal
Agent mode; incompatible combinations return validation errors. Drafting, if
separately enabled/authorized, still uses existing UNVERIFIED and human-review
rules. This acceptance used no paid drafting or scientific verifier.

## Failures and explicit fallback

- Disabled/missing runtime: 503. Missing index may return FALLBACK_HYBRID only if
  **both** server `GRAPH_RAG_ALLOW_HYBRID_FALLBACK` and request
  `fallback_to_hybrid` permit it. A fallback receipt never says USED.
- Unauthorized paper/seed: 403; bounds/unsupported filters: 422.
- Stale generation, changed canonical source, altered graph/Wiki or incompatible
  runtime schema: refused (409 or the existing D6 identity error); no fallback.
- Unexpected runtime exception: explicit unavailable error; no fabricated evidence.

## Reproduce the engineering acceptance

Create a **new empty local PostgreSQL database** whose name starts with
`graph_acceptance_`. Use an existing compatible local Qdrant instance. The script
creates a new collection and DATA_DIR; it does not mutate an existing corpus.
Use source files you are authorized to process; the repository distributes none.

```powershell
$env:GRAPH_ACCEPTANCE_DATABASE_URL="postgresql+psycopg://<user>:<password>@127.0.0.1:5432/graph_acceptance_new"
$env:PYTHONDONTWRITEBYTECODE="1"
python -B scripts/accept_graph_rag_runtime.py --pdf <authorized-paper-a.pdf> --pdf <authorized-paper-b.pdf> --query "Relate the method definition to its experimental configuration." --output .runtime/graph-acceptance-new
```

The output must not already exist. Test-server processes stop automatically;
the isolated corpus, collection and database are retained for inspection. The
script uses local hash embeddings, real PDF parsing, real HTTP Qdrant/PostgreSQL,
a real Uvicorn API, real Agent tools and exact PDF page checks; no mocked results
or paid calls. It does not qualify Qwen/CLIP performance, scientific multi-hop
reasoning, production load capacity or trusted report publication. See the
[integration report](integration_report.md) for actual results and limitations.
