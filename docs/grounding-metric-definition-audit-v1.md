# Grounding Metric Definition Audit v1

## Required Claim Coverage

- Current implementation: deterministic token-overlap heuristic.
- Embedding: not used.
- LLM judge: not used.
- Threshold: best generated-claim overlap >= `0.35`.
- One generated claim can satisfy multiple required claims mathematically because each required claim independently takes the best overlap across all generated claims.
- Multiple generated claims cannot jointly satisfy one required claim; the score is the maximum single-claim overlap.
- Composite claims are not semantically decomposed; this can under-credit partially correct composite answers and over-credit broad paraphrases.

## Citation Precision

The current metric checks whether each cited `(paper_id, page, block_id)` triple exactly matches the question-level Gold paper/page/block sets. It does not judge whether the citation semantically supports the generated claim.

Audited name: `gold_citation_exact_match_precision`.

## Citation Recall

The current metric measures cited Gold block coverage, i.e. `len(cited_gold_blocks) / len(gold_blocks)`. It is an exact Gold Block Recall, not semantic evidence recall.

Audited name: `gold_block_exact_recall`.

## Unsupported Claim

The current unsupported-claim count marks a generated claim unsupported when none of its citations exactly matches question-level Gold paper/page/block. It is deterministic and exact-Gold based. It is not an LLM judge and does not incorporate human citation labels or equivalent evidence unless those blocks/pages are in Gold.

Therefore, Gold-block mismatch must not be described as semantic unsupported without a separate human or semantic support audit.
