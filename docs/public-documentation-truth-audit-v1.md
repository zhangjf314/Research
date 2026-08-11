# Public Documentation Truth Audit v1

Audit scope: README, architecture docs, portfolio docs, release-readiness docs,
and linked public documentation after the v1.1.0 release.

Audit branch: `audit/v1.2.0-portfolio-readiness`

Baseline main commit:
`47372e76f33347675c65c3ad6fbbaaa4e778e991`

## Summary

| Check | Result |
| --- | --- |
| README local documentation links checked | 30 |
| Missing README local documentation links after this audit | 0 |
| Stage 4 benchmark rerun | `false` |
| Live provider/model calls | `0` |
| v1.2.0 version bump during readiness audit | `false` |
| v1.2.0 tag created during readiness audit | `false` |
| GitHub Release created during readiness audit | `false` |
| R1-R4 reliability branch merged | `false` |
| Workflow known reliability limitation documented | `true` |
| Agent final-report fresh live smoke attempted | `false` |

## Truth matrix

| Document | Linked | Purpose | Current or historical | Accurate after audit | Problems found | Action |
| --- | --- | --- | --- | --- | --- | --- |
| `README.md` | root | Public homepage | Current plus frozen historical benchmark | Yes | Needed clearer Workflow/Agent mode boundary and v1.2 candidate status. | Updated. |
| `docs/architecture.md` | README | System architecture | Current runtime | Yes | Previous file was stale and contained mojibake/over-broad graph claims. | Rewritten. |
| `docs/pdf-rag-data-flow.md` | README | PDF to RAG pipeline | Current runtime plus frozen backend | Yes | Previous file was stale and implied optional components as normal path. | Rewritten. |
| `docs/langgraph-workflow.md` | README | Deep Research Workflow | Current Workflow/control path | Yes | Needed explicit separation from Agent. | Rewritten. |
| `docs/research-agent/research-agent-runtime.md` | README | Agent runtime | Current runtime | Yes | Missing current runtime boundary doc. | Added. |
| `docs/portfolio/project-summary-v2.md` | README | Public project summary | v1.2 release summary | Yes | v1 materials did not capture post-v1.1 UI/Agent-report boundary. | Added v2. |
| `docs/portfolio/interview-notes-v2.md` | README | Interview-safe wording | v1.2 release summary | Yes | Needed safe claims and forbidden claims aligned to Stage 4 validity. | Added v2. |
| `docs/portfolio/release-status-v2.md` | README | Release status | v1.2 release status | Yes | Needed to state release status and documented limitations. | Added v2. |
| `docs/releases/v1.2.0-change-inventory.md` | README | Change inventory | Candidate readiness | Yes | Missing inventory for post-v1.1 changes. | Added. |
| `docs/releases/v1.2.0-version-truth-table.md` | release docs | Version source truth | Candidate readiness | Yes | Needed explicit package/runtime/display/tag separation. | Added. |
| `docs/releases/v1.2.0-portfolio-readiness.md` | README | Readiness decision | Candidate readiness | Yes | Missing readiness decision with limitations. | Added. |
| `docs/research-agent/benchmark/stage4-final-benchmark-v1.md` | README | Official Stage 4 result | Historical v1.1 artifact | Yes | Must not be reinterpreted with final-report synthesis. | Preserved. |
| `docs/research-agent/benchmark/stage4c-final-validity-audit-v1.md` | README | Validity audit | Historical v1.1 artifact | Yes | No action; boundaries referenced from README. | Preserved. |
| `docs/known-limitations.md` | README | Limitations | Current limitations with historical boundaries | Yes | Contained stale RC-era "current blocker" text that conflicted with later release evidence. | Rewritten. |
| `docs/releases/v1.2.0-portfolio.md` | README | Release notes | v1.2 release | Yes | Formal release note needed after authorization. | Added. |

## Architecture facts verified from source

- UI research mode defaults to `workflow` unless `mode=agent` is provided.
- Workflow and Agent call separate API endpoints:
  - Workflow: `POST /api/v1/research/deep`
  - Agent: `POST /api/v1/research/agent`
- Agent final-report synthesis is invoked after Agent completion and
  verification, not as an Agent retrieval/tool/replan step.
- `ParserRouter` supports explicit `grobid`, `docling`, `ocr`, and `pymupdf`
  backends plus `auto` routing.
- Redis is used for cache, rate limiting, import lock, and health/capability
  telemetry; it is not a retrieval backend.
- Stage 3 RAG backend lock keeps reranker, query rewrite, and query
  decomposition disabled.

## README local link audit

All README-local Markdown links resolve after this audit.

| Link | Exists |
| --- | --- |
| `docs/research-agent/benchmark/stage4-final-benchmark-v1.md` | yes |
| `docs/research-agent/benchmark/stage4c-final-validity-audit-v1.md` | yes |
| `docs/research-agent/benchmark/stage4c-metric-provenance-v1.md` | yes |
| `docs/research-agent/benchmark/stage4-portfolio-claim-boundary-v1.md` | yes |
| `docs/research-agent/benchmark/stage4-portfolio-release-readiness-v1.md` | yes |
| `docs/quickstart.md` | yes |
| `docs/api-examples.md` | yes |
| `docs/architecture.md` | yes |
| `docs/pdf-rag-data-flow.md` | yes |
| `docs/langgraph-workflow.md` | yes |
| `docs/research-agent/research-agent-runtime.md` | yes |
| `docs/deployment-runbook.md` | yes |
| `docs/docker-ocr-production-audit-v2.md` | yes |
| `docs/langgraph-production-recovery-audit-v2.md` | yes |
| `docs/backup-restore-audit.md` | yes |
| `docs/portfolio-evaluation-policy-v1.md` | yes |
| `docs/deepseek-full-qa-final-summary-v1.md` | yes |
| `docs/end-to-end-deepseek-production-v2.md` | yes |
| `docs/portfolio/project-summary-v2.md` | yes |
| `docs/portfolio/interview-notes-v2.md` | yes |
| `docs/portfolio/release-status-v2.md` | yes |
| `docs/git-history-secret-review-v1.md` | yes |
| `docs/known-limitations.md` | yes |
| `docs/releases/v1.2.0-portfolio.md` | yes |
| `docs/public-documentation-truth-audit-v1.md` | yes |
| `docs/releases/v1.2.0-change-inventory.md` | yes |
| `docs/releases/v1.2.0-version-truth-table.md` | yes |
| `docs/releases/v1.2.0-portfolio-readiness.md` | yes |

## Release claim boundary

Public materials may say:

> PaperResearch includes a frozen Deep Research Workflow baseline, an explicit
> Research Agent mode, verified Evidence State, strict citation validation,
> replayable observability, and a Stage 4 structured paired benchmark with
> documented limitations.

Public materials must not say:

> The Stage 4 benchmark proves strong semantic grounding, large-scale blind
> generalization, commercial production readiness, or hallucination elimination.
