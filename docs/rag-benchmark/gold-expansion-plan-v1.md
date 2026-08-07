# RAG Gold expansion plan v1

This plan expands the current human-reviewed internal Gold set toward approximately 150 questions before any Stage 2 RAG optimization.

- current_total: 50
- target_total: 150
- questions_to_add: 100
- candidate review_status default: `pending`
- approved requires human review: `true`

## Category deficits

| category | current | target | deficit |
| --- | ---: | ---: | ---: |
| single_hop_factual | 25 | 30 | 5 |
| multi_evidence_synthesis | 0 | 30 | 30 |
| cross_paper_comparison | 2 | 30 | 28 |
| methods_and_experiments | 14 | 25 | 11 |
| limitations_and_research_gaps | 7 | 20 | 13 |
| unanswerable | 2 | 15 | 13 |

## Difficulty guidance

- easy: 25-30%
- medium: 40-50%
- hard: 25-30%

New questions must not all be simple factual questions. Medium and hard questions should require multiple evidence blocks, multiple sections, cross-paper comparison, conflicting evidence, methods/results comparison, or limitations synthesis.

## Corpus coverage

- corpus_paper_count: 33
- papers_covered_by_current_gold: 10

## Review policy

LLM or rule-assisted authoring may only create draft candidates. Final Gold questions, required claims, answers, pages, and block evidence must be human reviewed before `review_status=approved`.
