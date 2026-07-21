# Artifact Publication Audit v1

Status: `PUBLICATION_RESTRICTED`

## Safe to publish

- `data/evaluation/deepseek-full-qa-final-v1.json`
- `data/evaluation/deepseek-full-qa-final-v1.csv`
- `docs/deepseek-full-qa-final-summary-v1.md`
- `docs/deepseek-full-qa-final-audit-v1.md`
- `docs/end-to-end-deepseek-production-v2.md`
- `data/evaluation/portfolio-evidence-manifest-v1.json`
- `docs/portfolio-evidence-manifest-v1.md`

## Keep local-only

- `artifacts/stage13-9-human-citation-review-results.zip`
- `artifacts/stage13-10-human-claim-gold-review-results.zip`
- `artifacts/deepseek-full-qa-final-trace-v1.json`
- `artifacts/deepseek-production-deep-research-v1/`
- `artifacts/evidence-first-canary-trace-v1.json`
- `artifacts/full-qa-canary-deepseek-trace-v1.json`
- `artifacts/full-qa-canary-trace-v2.json`
- `artifacts/full-qa-rerun-backups/`
- `artifacts/stage13-31-q002-provider-response-audit.json`
- `artifacts/stage13-31-q002-reproduction-input.json`
- `data/evaluation/deepseek-full-qa-final-items-v1.jsonl`
- `data/evaluation/evidence-first-canary-v1.*`
- `data/evaluation/full-qa-canary-results-v2.*`
- `docs/full-qa-canary-audit-v2.md`

Reason: these files may include raw model outputs, long context excerpts,
provider payloads, item-level answers, or intermediate debugging data. They are
useful for local auditability but should not be included in a public portfolio
release without separate redaction.
