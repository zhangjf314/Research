# Content Claims Audit v1

Status: `PASSED_WITH_LIMITATIONS`

Allowed public wording:

> Based on 50 human-reviewed internal development evaluation records, the
> project completed retrieval and QA evaluation with real DeepSeek model calls.

Allowed with caveat:

> A 27-claim diagnostic benchmark was used for retrieval and citation failure
> analysis.

Forbidden wording:

- strict blind holdout
- large-scale independent benchmark
- production-grade generalization proven
- strong semantic grounding proven
- production-ready v1.0
- `v1.0.0-portfolio` release-ready
- stability claims beyond the measured 30-minute window
- commercial endurance equivalence claims

Required caveats:

- `gold-dev-v1` is an internal development evaluation set, not a blind holdout.
- `retrieval-diagnostic-v1` is diagnostic and has been used during development.
- `shadow-holdout-pilot-v1` has not been created.
- `STRONG_GENERALIZATION_CLAIM_ALLOWED=false`.
- `SEMANTIC_CLAIM_SUPPORT_AUDIT=NOT_FORMALLY_VALIDATED`.
- Full QA passed the engineering gate, but exact citation/claim metrics remain
  modest and should be described as measured limitations, not hidden.
- The Portfolio 30-minute stability test is a bounded engineering check. If it
  passes, only say that no obvious sustained abnormal memory growth was observed
  within that 30-minute window.
