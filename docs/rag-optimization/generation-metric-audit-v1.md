# Generation metric audit v1

- status: `METRICS_VALID`
- answerable_questions: `132`
- citation_precision: `0.0` (0/430)
- citation_recall: `0.097973` (29/296)

## Representation

- gold: paper_id + page + block_id from gold_paper_ids/gold_pages/gold_block_ids
- generated: claim.citations[].paper_id/page/block_id
- paper_id_normalization: exact string comparison
- block_id_normalization: exact string comparison
- page_normalization: integer comparison

## Explanation

Precision can be 0 while recall is positive because precision requires an exact paper/page/block triple for every generated citation, while recall only counts whether any generated citation mentions a gold block ID. In the frozen results, some answers cite gold block IDs but fail exact page or paper matching.

## Sample buckets

- no_citation: `30`
- wrong_citation: `102`

## Sampled examples

- `q001`: no_citation, correct=0/0
- `q002`: no_citation, correct=0/0
- `q003`: no_citation, correct=0/0
- `q004`: no_citation, correct=0/0
- `q006`: wrong_citation, correct=0/6
- `q007`: wrong_citation, correct=0/6
- `q008`: wrong_citation, correct=0/4
- `q009`: wrong_citation, correct=0/6
- `q010`: wrong_citation, correct=0/5
- `q011`: no_citation, correct=0/0
