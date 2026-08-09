# Portfolio Release Audit v1

Final portfolio release conclusion:

`A. v1.1.0-portfolio release finalization is authorized with READY_WITH_SEMANTIC_EVALUATION_LIMITATION.`

## Current evidence

- Branch before final merge: `eval/research-agent-benchmark-run-v1`
- Version: `1.1.0+portfolio` / display `1.1.0-portfolio`
- Previous tags retained: `v1.0.0-portfolio`, `v1.0.1-portfolio`
- Stage 4 release readiness: `READY_WITH_SEMANTIC_EVALUATION_LIMITATION`
- Provider/model: `deepseek` / `deepseek-v4-flash`
- Reranker enabled: `false`
- Template fallback: `false`

## Passed engineering evidence

- DeepSeek Full QA: `50/50` completed, `0` failed.
- Full QA total tokens/cost: `529410` / `$0.07508382`.
- Deep Research q003 final run: `completed`, `6796` tokens, `$0.00096292`.
- Strict citation validation: passed for the final Deep Research run.
- PostgreSQL checkpoint recovery v2: `PASSED`.
- PostgreSQL backup/restore v2: `PASSED`.
- Qdrant snapshot/restore v2: `PASSED`.
- Docker OCR text/mixed/scanned roundtrip v2: `PASSED`.
- Git history secret review: `PASSED`, with `0` confirmed real secrets and `0`
  unresolved hits.
- Portfolio 30-minute stability test: `PASSED`.

## Portfolio 30-minute stability evidence

- Actual duration: `1802.453` seconds.
- Requests: `568`; failures: `0`.
- Fatal errors: `0`; unclassified exceptions: `0`.
- Latency P95: `4846.271` ms.
- API restart count: `1`; recovery: `passed` in `2.113` seconds.
- Short QA: `3/3`, success rate `1.0`.
- Short Deep Research success count: `1`.
- OCR roundtrip during stability: `passed`.
- Total tokens/cost: `2481` / `$0.00006328`.
- Active reserved tokens: `0`.
- Redis cache hit rate: `0.929791`.

Allowed memory interpretation:

> Within this 30-minute test window, no obvious sustained abnormal memory growth was observed.

Forbidden interpretations remain: no memory leak, long-term stable,
production-grade endurance passed, or commercial production ready.

## Remaining semantic limitations

These are not release-blocking portfolio engineering gates, but they constrain
public wording:

- `SEMANTIC_CLAIM_SUPPORT_AUDIT=NOT_FORMALLY_VALIDATED`
- `STRONG_GROUNDING_CLAIM_ALLOWED=false`
- `RETRIEVAL_GENERALIZATION_EVIDENCE=DIAGNOSTIC_ONLY`
- No large independent blind benchmark was run.

## Release boundary

The final release process may merge the release branch into `main`, create the
annotated tag `v1.1.0-portfolio`, and push `main` plus that tag after the final
test gates pass. This does not create a GitHub Release and does not move any
existing tag.
