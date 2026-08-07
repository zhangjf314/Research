# RAG Gold expansion plan v1

This plan expands the current human-reviewed internal Gold set toward approximately 150 questions before any Stage 2 RAG optimization.

- current_total: 146
- target_total: 150
- questions_to_add: 4
- candidate review_status default: `pending`
- approved requires human review: `true`

## Category deficits

| category | current | target | deficit |
| --- | ---: | ---: | ---: |
| single_hop_factual | 30 | 30 | 0 |
| multi_evidence_synthesis | 30 | 30 | 0 |
| cross_paper_comparison | 27 | 30 | 3 |
| methods_and_experiments | 25 | 25 | 0 |
| limitations_and_research_gaps | 20 | 20 | 0 |
| unanswerable | 14 | 15 | 1 |

## Difficulty guidance

- easy: 25-30%
- medium: 40-50%
- hard: 25-30%

New questions must not all be simple factual questions. Medium and hard questions should require multiple evidence blocks, multiple sections, cross-paper comparison, conflicting evidence, methods/results comparison, or limitations synthesis.

## Corpus coverage

- corpus_paper_count: 33
- papers_covered_by_current_gold: 33

## Review policy

LLM or rule-assisted authoring may only create draft candidates. Final Gold questions, required claims, answers, pages, and block evidence must be human reviewed before `review_status=approved`.
