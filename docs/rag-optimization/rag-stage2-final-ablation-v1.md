# RAG Stage 2 Final Ablation

## Scope

Stage 2 finalized predefined RAG optimization experiments. It did not add new retrievers, rerankers, rewrites, context selectors, prompts, embeddings, models, or agent behavior.

## Experimental Protocol

- dataset: `rag-gold-v1`
- dev hash: `f61fc199c559250d32811f755db1400114b131b5da7b94b7abab6be0340c722a`
- test hash: `e991feb4d1d60a852926d736ed4e0a97f72a437b67def5dfb9afe4cde4e0eaf8`
- Stage 2 split: `dev` only
- new TEST runs: `0`
- new provider requests in Stage 2D: `0`

## Retrieval Ablation Table

| Configuration | R@10 | MRR@10 | nDCG@10 | EvidenceCov@10 | ReqClaimEvidenceCov@10 | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Dense Only | 0.396212 | 0.206029 | 0.222883 | 0.396212 | N/A | REJECTED |
| Sparse Only | 0.452652 | 0.199220 | 0.237179 | 0.452652 | N/A | REJECTED |
| Current Hybrid | 0.469697 | 0.219467 | 0.250609 | 0.469697 | 0.561321 | SELECTED |
| Hybrid + Lexical Rerank | 0.473864 | 0.284776 | 0.276777 | 0.473864 | N/A | REJECTED |
| Single Rewrite | 0.455871 | 0.213515 | 0.239102 | 0.455871 | 0.528302 | REJECTED |
| Original + Rewrite | 0.325947 | 0.166797 | 0.279072 | 0.325947 | 0.433962 | REJECTED |
| Original + Decomposition | 0.340720 | 0.164002 | 0.252639 | 0.340720 | 0.438679 | REJECTED |

## Context Ablation Table

| Configuration | ReqClaim Context Cov | Full Context Cov | P95 Tokens | Decision |
| --- | ---: | ---: | ---: | --- |
| Baseline Context | 0.358491 | 0.272727 | 4160.000000 | BASELINE_RETAINED |
| Score-Budgeted Dedup | 0.363208 | 0.284091 | 4160.000000 | REJECTED |
| Diversity-Aware | 0.292453 | 0.204545 | 4080.000000 | REJECTED |

Context Selection Bottleneck: `CONFIRMED`.

Effective Selector: `NOT FOUND IN PREREGISTERED EXPERIMENTS`.

## Component Decision Matrix

| Component | Hypothesis | Evidence | Gate | Decision |
| --- | --- | --- | --- | --- |
| Hybrid | Dense/Sparse complementary | Supported | Passed | SELECTED |
| Lexical Rerank | Deep candidates can be promoted | Partial | Failed | REJECTED |
| Single Rewrite | Better search formulation improves recall | Not supported | Failed | REJECTED |
| Multi-query Rewrite | Original plus rewrite improves retrieval | Contradicted | Failed | REJECTED |
| Query Decomposition | Complex questions benefit from decomposition | Contradicted | Failed | REJECTED |
| Context Selection Bottleneck | Evidence is lost between retrieval and final context | Confirmed | Diagnostic | SUPPORTED_BUT_UNRESOLVED |
| Score Context | Deduplication reduces evidence loss | Weak | Failed | REJECTED |
| Diversity Context | Diversity improves multi-paper evidence | Contradicted | Failed | REJECTED |

## Final RAG Configuration

- final retriever: `Current Hybrid`
- final reranker: `Disabled`
- final query rewrite: `Disabled`
- final query decomposition: `Disabled`
- final context selector: `Baseline`
- behavior change: `False`
- final config hash: `995a144385180b2931ec2c6366f7f7306301a42d77ad7c85f4be9e6d9e5091d9`

## Held-out Test Status

- source: `REUSED_STAGE1_FROZEN_BASELINE`
- new_test_questions_evaluated: `0`

Because the final Stage 2 behavior is equivalent to the Stage 1 frozen baseline behavior, Stage 2 reuses the Stage 1 frozen TEST baseline instead of rerunning 48 TEST retrieval/generation cases.

## Negative Findings

### Lexical Rerank

MRR improved, but retrieval coverage did not materially improve enough to meet the preregistered recall/evidence-coverage gate.

### Query Rewrite

All preregistered rewrite strategies failed to beat the untouched Hybrid baseline. Q2/Q3 substantially reduced Recall@10.

### Context Selector

The evidence-drop bottleneck is real, but simple deterministic dedup/diversity heuristics did not recover enough required-claim evidence.

## Remaining Bottlenecks

- Retrieval miss: `UNRESOLVED`
- Context evidence drop: `CONFIRMED_UNRESOLVED`
- Generation utilization: `CONFIRMED_DIAGNOSTIC`
- Citation exactness: `UNRESOLVED`

## Stage 2 Limitations

- Stage 2B exact rewrite cost is unavailable because the first sanitized cache schema did not persist complete provider usage.
- Stage 2C context traces are deterministic reconstructions, not original generation-time telemetry.
- Stage 2C `SUCCESS=0` means strict full-question success; it does not mean all 88 answerable questions were entirely wrong.

## Stage 3 Handoff

Stage 3 must use the Stage 2 frozen Hybrid backend. It must not automatically enable rejected Stage 2 components.

## Final Conclusion

Stage 2 experimentally evaluated four predefined optimization families. Hybrid retrieval was validated and retained. Lexical reranking improved rank-sensitive metrics but failed the preregistered recall/evidence gate. Query rewriting and decomposition regressed retrieval quality. Context selection was confirmed as a major evidence-loss stage, but the two preregistered deterministic selectors did not pass the offline gate. No unsupported component was promoted into the final system.
