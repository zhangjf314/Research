# Known Limitations

This document records current public limitations. Older RC-stage limitations
remain in Git history and stage-specific reports, but they must not be treated
as current facts when they conflict with later release evidence.

## v1.2.0 candidate boundary

- The current package/runtime version remains `1.1.0+portfolio` until a separate
  release commit is authorized.
- This readiness audit does not create a v1.2.0 tag, GitHub Release, or version
  bump.
- This readiness audit does not rerun Stage 4, Full QA, Deep Research, or any
  live provider/model task.
- Stage 4 Workflow-vs-Agent benchmark artifacts remain frozen v1.1.0 evidence.
  Agent final-report synthesis is a current runtime feature, but it is not part
  of the Stage 4 benchmark.

## Workflow reliability limitation

- The historical Deep Research Workflow has known reliability risk around
  provider schema, report quality, and strict synthesis validation.
- This limitation blocks production-reliability claims.
- It does not block a truthful portfolio release when the limitation is clearly
  disclosed.
- The older v1.1.1 R1-R4 reliability patch series is not merged into current
  `main` and must not be claimed as released behavior.

## Research Agent benchmark limitations

- Stage 4 benchmark tasks were internally authored and reviewed.
- Stage 4 is not a public benchmark and not a strict blind generalization
  benchmark.
- `budget_comparable=false`; the Workflow-vs-Agent result is not a strict
  equal-budget causal ablation.
- Structured claim/dimension/evidence coverage metrics are structural proxies,
  not human semantic rubric scores over every claim.
- Effective live replan was not observed in the final Stage 4 benchmark:
  `LIVE_EFFECTIVE_REPLAN_NOT_OBSERVED`.
- Agent final-report synthesis was added after the frozen Stage 4 benchmark and
  must not be used to reinterpret Stage 4 scores.

## Portfolio evaluation limitations

- `gold-dev-v1` contains 50 human-approved records and is an internal
  development evaluation set. It is not a blind holdout, public benchmark, or
  strict generalization benchmark.
- `retrieval-diagnostic-v1` contains 27 claim-level records used for diagnostic
  failure analysis and regression checks. It has been inspected during
  development and must not be described as blind.
- `shadow-holdout-pilot-v1` has not been created. It is recommended as a
  10-15-sample small blind pilot, but it is not required for portfolio Full QA.
- `RETRIEVAL_GENERALIZATION_EVIDENCE=DIAGNOSTIC_ONLY`; strong generalization
  claims are not allowed.
- Public materials must disclose that there is no large-scale independent blind
  benchmark result.

## Deep Research report boundary

- The post-v1.0.1 Deep Research report path uses section-specific retrieval,
  deduplicated evidence cataloging, section-scoped citation allowlists,
  structured synthesis, deterministic Markdown generation, and report quality
  gates.
- The report quality gate validates duplicate text, duplicate references,
  citation ID presence, and catalog consistency. It does not prove full semantic
  claim support.
- `SEMANTIC_CLAIM_SUPPORT_AUDIT=NOT_FORMALLY_VALIDATED` and
  `STRONG_GROUNDING_CLAIM_ALLOWED=false` remain in force.

## Operations and deployment limitations

- The Docker Compose stack is a local portfolio deployment. It includes local
  development database credentials and is not hardened for direct exposure to an
  untrusted network.
- The Portfolio 30-minute stability test supports only this statement: "Within
  this 30-minute test window, no obvious sustained abnormal memory growth was
  observed." It must not be described as proof of long-term stability or a
  commercial endurance validation.
- Optional parser capabilities depend on host/container configuration: Docling
  requires the optional dependency, GROBID requires a configured reachable
  service, and OCR requires Tesseract.
- The application does not claim a complete public-edge security boundary,
  multi-tenant isolation, or commercial production SLOs.
