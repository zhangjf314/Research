# Stage 11A — Real Embedding and Retrieval Ablation

## Scope and current status

- Branch: `eval/real-embedding-v1`
- Gold set: 50/50 `approved`; pure retrieval metrics evaluate the 48 answerable records.
- Baseline: Hash 384-dimensional embedding.
- Production target: `jina-embeddings-v5-text-small`, 1024 dimensions.
- Reranker: disabled for every Stage 11A variant.
- LLM: Template remains configured; the ablation runner never constructs or calls an LLM.
- Deep Research: not run.

Stage 11A real execution completed on 2026-07-13. The live Docker corpus contained 36
papers rather than the planned 34 because two retained Stage 10 OCR fixtures were also
present. They were not deleted or excluded after discovery.

The official Jina model description specifies 1024 output dimensions and asymmetric
retrieval encoding. The provider sends `retrieval.query` for queries and
`retrieval.passage` for indexed documents. See the
[Jina v5-small model page](https://jina.ai/models/jina-embeddings-v5-text-small/) and
[Jina embedding API](https://jina.ai/en-US/embeddings/).

## Configuration

Public configuration fields:

```text
APP_PROFILE=production
EMBEDDING_PROVIDER=jina
EMBEDDING_MODEL=jina-embeddings-v5-text-small
EMBEDDING_REVISION=<pinned revision or release label>
EMBEDDING_BASE_URL=https://api.jina.ai/v1
EMBEDDING_API_KEY=<local secret>
EMBEDDING_DIMENSIONS=1024
EMBEDDING_BATCH_SIZE=32
EMBEDDING_TIMEOUT_SECONDS=60
EMBEDDING_MAX_RETRIES=2
RERANK_ENABLED=false
LLM_PROVIDER=template
BASELINE_COLLECTION=papers_hash_v1
PRODUCTION_COLLECTION=papers_production_v1
```

Production missing provider, key, model, or dimensions fails explicitly. It never
selects Hash as a fallback. Error messages contain only sanitized exception types or HTTP
status codes, not response bodies, request headers, vectors, or API keys.

## Collection rules

- Existing Hash logical name: `papers_hash_v1`.
- Production logical name: `papers_production_v1`.
- Jina physical names follow
  `papers_jina_v5_text_small_1024__<index_version>__<timestamp>`.
- The Qdrant collection is created with the configured provider dimension before upsert.
- Registry switching occurs only after all papers are indexed successfully.
- A failed staging collection is retained and reported; Stage 11A does not delete it
  automatically.
- Existing Hash and restore collections are never deleted or overwritten.

Registry records logical/physical names, provider, model, revision, dimension, structural
chunker, index version, creation time, paper count, point count, build duration, and
status.

## Connectivity check

`scripts/check_embedding_provider.py` embeds one query and two short documents. It prints
only provider/model/revision, expected dimension, vector counts, and elapsed time. It
never prints vectors or credentials and exits non-zero on provider, count, or dimension
failure.

## Retrieval ablation

`scripts/run_retrieval_ablation_v1.py` runs:

1. `hash_structural_dense`
2. `hash_structural_hybrid`
3. `jina_structural_dense`
4. `jina_structural_hybrid`

All variants use fixed Top-10 output and recall depth 20. Outputs are:

- `data/evaluation/retrieval-ablation-v1.json`
- `data/evaluation/retrieval-ablation-v1.csv`
- `docs/retrieval-ablation-v1.md`

The JSON contains overall, category, and difficulty metrics plus every ranked result,
gold paper/block/page hits, latency, and failure reason. Report conclusions must be based
on measured results; the runner does not assume the real model improves the baseline.

## Safe execution boundary

Offline implementation and mock tests must pass first. When a real key is required, stop
and let the user perform:

```powershell
Copy-Item .env.stage11a.local .env -Force
```

The user should fill `EMBEDDING_API_KEY` locally and never send it through chat or logs.
Then rerun:

```powershell
python scripts\check_embedding_provider.py
```

Only after that command reports a real Jina 1024-dimensional PASS should the Production
collection be rebuilt and the four-way ablation executed.

## Real execution audit

The safe connectivity check passed with one query and two documents:

- provider: `jina`
- model: `jina-embeddings-v5-text-small`
- revision: `v1`
- dimension: 1024
- elapsed: 2410.494 ms

The first rebuild (`16495aa6-1d41-4642-8c44-d397cc5ed9a8`) was correctly reported as
`FAILED_ROLLED_BACK` after HTTP 429. It completed 6/36 papers, retained its 325-point
staging collection, and did not activate the Production logical collection. This run
exposed that retry handling needed to honor Jina rate-limit reset headers.

After that fix, rebuild `d15e860d-8506-4599-8b31-548c9f7371db` completed:

- papers: 36/36
- points: 2064
- build duration: 475.512 seconds
- active physical collection:
  `papers_jina_v5_text_small_1024__jina_embeddings_v5_text_small_v1__20260713144312`
- logical collection: `papers_production_v1`
- status/dimension: green, 1024, Cosine

The retained baseline is
`papers_hash_v1__20260713104355` (green, 384 dimensions, 2065 points). The failed Jina
staging collection and all older Hash/restore collections also remain present. The
one-point difference reflects the current full rebuild versus the older baseline's
stored corpus state; no Hash point or collection was modified by Stage 11A.

## Measured conclusion

The formal pure-retrieval run used all 48 approved answerable questions; the two approved
unanswerable records were excluded from retrieval metrics. No query failed, no LLM was
called, and reranking stayed disabled. See `docs/retrieval-ablation-v1.md` and the JSON
artifact for complete results.

On this run Jina underperformed Hash for both Dense and Hybrid retrieval. In the Hybrid
comparison, MRR changed from 0.158 to 0.082, NDCG@10 from 0.191 to 0.131, Recall@5 from
0.208 to 0.125, and mean latency from 23.645 ms to 436.747 ms. Jina therefore must not
become the Production default from Stage 11A evidence.

The result also has a protocol limitation: many questions say only `the target paper`
without identifying that paper. Such queries are ambiguous in corpus-wide retrieval.
Stage 11B should first make questions independently identifiable or supply an explicit
non-gold scope constraint, then rerun the same four variants before drawing a general
model-quality conclusion.

## Stage 11A.5 superseding note

Stage 11A.5 subsequently implemented that protocol correction. The v1 metrics above
remain historical evidence and were not overwritten. The scope-aware v2 result uses a
fixed 34-document corpus and reports known-paper Block retrieval separately from
multi-paper and unanswerable behavior. Under that corrected task, Jina Hybrid exceeds
Hash Hybrid on the paper-scoped acceptance metrics and is now an embedding candidate,
not yet an enabled Production default. The two rewritten unanswerable queries were
human-approved by `zjf` on 2026-07-13. See `docs/retrieval-ablation-v2.md`; Stage 11B may
now begin with Reranker still disabled by default.
