# v1 Release Gates

The authoritative definition is
[`data/evaluation/v1-release-gates.json`](../data/evaluation/v1-release-gates.json). Every gate has
an ID, threshold, source, current value, status, blocker type, remediation, and separate RC/v1
requirements.

## Current decision

- Existing tags: `v0.9.0-rc1`, `v0.9.0-rc2`.
- Proposed next candidate: `v0.9.0-rc3`.
- Package metadata is aligned to `0.9.0rc3`; no Tag is created by Stage 12.
- RC scope permits disclosed quality limitations and external blockers; it never permits a
  Production-ready claim.
- v1 requires every `required_for_v1` gate to pass. It currently fails.

## Gate summary

| Area | Passing evidence | Open v1 gates |
|---|---|---|
| Data/corpus | 34-document manifest, two OCR fixtures isolated, 2062 points, Gold 50/50, Retrieval Gold 50/50 | None in the defined data gates |
| Retrieval | Corrected protocol, versioned Hash/Jina results, category/difficulty reporting | Exact block availability 0.416667; page availability 0.645833; multi-paper exact evidence |
| Reranking | Strict provider behavior, enablement rule, negative result retained | None while `RERANK_ENABLED=false` |
| QA | JSON/schema 1.0, refusal accuracy 1.0, Token/latency recorded | Answerable 0.875; required claims 0.388889; exact citation precision 0.103009; recall 0.096875; human support |
| Deep Research | Three bounded outcomes, budgets, checkpoints, isolation and failure accounting | Successful resumed completion, quality, citation quality, convergence quality |
| Operations | Docker acceptance, config validation, secret scan, migration, backup/restore, traces, 148-test current-tree report | Qdrant version compatibility and HTTP/API-key transport warning |

Citation ID validity is structural integrity only. It cannot satisfy citation quality. The
30-sample audit is AI-assisted and failure-stratified; it is not independent double-blind review
and cannot be extrapolated as full-dataset precision. Oracle diagnostics are never Production
metrics.

## Version levels

| Version | Real models | Human Gold | QA quality | Deep Research quality | External blockers | Production-ready |
|---|---|---|---|---|---|---|
| v0.9.0-rc1 | Not required | Not required | Not required | Not required | Allowed if disclosed | No |
| v0.9.0-rc2 | Not required | Required | Not required | Not required | Allowed if disclosed | No |
| proposed v0.9.0-rc3 | Embedding and LLM required; reranker may remain disabled | Required | Negative results accepted and disclosed | Engineering smoke only | Allowed if disclosed | No |
| v1.0.0 | Required Production providers | Required | Required | Required | No unresolved release blockers | Yes |

Run:

```powershell
.\.venv\Scripts\python.exe scripts\check_release_readiness_v1.py --target rc --strict
.\.venv\Scripts\python.exe scripts\check_release_readiness_v1.py --target v1 --strict
```

A nonzero v1 exit is the expected release decision while the listed gates remain open; it is not
a checker malfunction.
