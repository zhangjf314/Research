# Stage 11B: real Jina Reranker

## Scope and current status

Stage 11B changes only the Reranker. Embedding, chunks, the 34-document Production
corpus, `retrieval-gold-v2`, filters, query order, and retrieval settings remain fixed.
The LLM remains the template provider and Deep Research is not executed.

The offline provider, trace, ablation runner, and mock tests are implemented. Real API
connectivity and the three-way ablation completed on 2026-07-14. The engineering
adapter passed, but the measured quality/latency acceptance conditions do not support
enabling the Jina Reranker by default.

## Fixed protocol

The formal comparison uses Jina `jina-embeddings-v5-text-small` with Structural Hybrid
retrieval against the existing 34-document, 2,062-point evaluation collection. Each
query is retrieved exactly once and the same immutable candidate snapshot is supplied
to all variants:

1. `no_rerank`
2. `lexical_rerank`
3. `jina_reranker_v3`

The ranking path is:

```text
Structural Hybrid retrieve Top-30
  -> rerank the same Top-30 candidate set
  -> retain the complete Top-30 ranking trace
  -> evaluate Top-10
```

The Reranker never scans Qdrant and the ablation runner does not create, update, or
delete collections. Formal runs require `RERANK_ALLOW_FALLBACK=false`; API failures are
counted and cannot silently switch to the lexical implementation.

## Provider contract

The Jina adapter calls `https://api.jina.ai/v1/rerank` with model
`jina-reranker-v3`. It validates the candidate count, output count, index bounds,
duplicate indexes, and finite relevance scores. Returned indexes are mapped back to the
original candidate objects so chunk, paper, dense-rank, sparse-rank, and other metadata
are preserved.

Errors expose only a sanitized status/reason and request count. Neither the API key nor
the response body is included in exceptions, traces, reports, or console output.

Configuration:

```dotenv
APP_PROFILE=production
RERANK_ENABLED=true
RERANK_PROVIDER=jina
RERANK_MODEL=jina-reranker-v3
RERANK_API_KEY=
RERANK_BASE_URL=https://api.jina.ai/v1
RERANK_INPUT_K=30
RERANK_OUTPUT_K=30
RERANK_TIMEOUT_SECONDS=60
RERANK_MAX_RETRIES=2
RERANK_ALLOW_FALLBACK=false
LLM_PROVIDER=template
```

When Production reranking is enabled, a missing key or model is a configuration error.
The lexical Reranker remains available only as an explicit baseline.

## Trace fields

Each query records retrieval scope and filters, pre-rerank candidate count, provider,
model, output count, latency, request count, fallback status, and sanitized failure
reason. All Top-30 candidates record pre-rerank rank and score, Reranker score, and
post-rerank rank.

## Offline verification

Run without a real model call:

```powershell
$env:APP_PROFILE='baseline'
$env:RERANK_ENABLED='false'
python -m pytest
python -m ruff check .
python -m compileall -q src scripts
git diff --check
```

## Real verification handoff

Keep the credential only in the ignored local environment file. Do not paste it into a
terminal command, report, test fixture, or Git-tracked file.

```powershell
Copy-Item .env.stage11b.local .env -Force
```

After `RERANK_API_KEY` is filled locally, run:

```powershell
python scripts\check_reranker_provider.py
python scripts\run_reranker_ablation_v1.py
```

The runner writes:

- `data/evaluation/reranker-ablation-v1.json`
- `data/evaluation/reranker-ablation-v1.csv`
- `docs/reranker-ablation-v1.md`

The final recommendation is based on real paper-scoped and multi-paper metrics,
pre/post Recall@10, latency, failures, fallbacks, and the distribution of improved and
regressed queries. `RERANK_ENABLED` remains false unless that evidence supports enabling
it.

## Real run result

The formal run used collection `papers_jina_eval34_v2__20260713152149`: 34 papers,
2,062 points, Jina embedding dimension 1,024, and chunk signature
`07f23a3a7456e61ac92467ab6cc65f5a0d8cae5cbd9877718879010d8b8d5fb2`.

| Variant | Paper Hit@1 | Paper Recall@10 | Paper MRR | Paper NDCG@10 | Total P95 ms |
|---|---:|---:|---:|---:|---:|
| no rerank | 0.152 | 0.307 | 0.245 | 0.158 | 699.3 |
| lexical | 0.130 | 0.309 | 0.231 | 0.154 | 704.6 |
| Jina v3 | 0.152 | 0.318 | 0.243 | 0.155 | 63,661.6 |

Jina completed with zero failed queries and zero fallbacks. It made 55 API requests,
including retries, for the 50-query protocol. Eleven answerable queries improved,
fourteen regressed, and twenty-three were unchanged. Although Recall@10 increased,
Hit@1 did not improve, MRR and NDCG@10 were slightly lower, and tail latency exceeded
the demo threshold. The resulting recommendation is to keep
`RERANK_ENABLED=false`. Full per-query rankings and metrics are in
`data/evaluation/reranker-ablation-v1.json` and `docs/reranker-ablation-v1.md`.
