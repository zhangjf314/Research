# Gold Set v1 Human Review Guide

`data/evaluation/gold-set-v1.jsonl` is a 50-item review queue. Its filename is the planned release artifact name; it is not a human gold set yet. Every generated item starts with `review_status=pending`.

## Required review procedure

1. Open each paper PDF and locate every `gold_page` and `gold_block_id` in the parse artifact.
2. Verify that the question is clear, scoped, and answerable from the cited paper(s).
3. Rewrite `gold_answer` and `required_claims` in the reviewer's own words where necessary. Do not approve an analyzer/LLM answer merely because it sounds plausible.
4. Verify each claim has a supporting block and page. Remove over-broad or irrelevant evidence.
5. For `multi_paper_comparison`, require evidence from every listed paper.
6. For `unanswerable`, search the full PDF and approve only if the requested exact information is absent; `gold_paper_ids`, blocks, and pages remain empty.
7. Set `review_status=approved` only after PDF inspection. Use `rejected` for an unusable item and keep `pending` when uncertain.

## Field contract

- `question_id`: stable ID.
- `scope`: `single_paper` or `multi_paper`.
- `category`: one of the documented coverage categories.
- `difficulty`: `easy`, `medium`, or `hard` pending reviewer confirmation.
- `answerable`: expected answer/refusal class.
- `gold_*`: human-confirmed paper, block, page, and answer fields after approval.
- `required_claims`: atomic points an answer must cover.
- `citation_notes`: reviewer instructions or known ambiguity.
- `review_status`: `pending`, `approved`, or `rejected`.

## Acceptance rule

Formal gold metrics may be reported only on `approved` items. Until at least two independent checks or one check plus adjudication are complete, RC reports must label metrics provisional.
