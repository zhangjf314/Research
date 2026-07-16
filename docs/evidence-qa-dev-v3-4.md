# Evidence QA Dev v3.4

## Outcome

- Provider completed: 10/10
- Raw JSON / structural / canonical: 10/10, 1/10, 1/10
- Final schema / slots: 1/10, 0/27
- Requests / tokens / cost: 10, 20078, USD 0
- Elapsed total / P50 / P95: 88.867s / 9.124s / 12.510s
- Failure mode: nine answerable responses used status values outside the frozen enum (`supported`, or `answerable` for q013); q001 also omitted the required top-level `refusal_reason`.
- q005 completed the strict unanswerable protocol with no claims or citations.

## Conservative quality metrics

- Question macro exact relation recall: 0.000000
- Required-claim macro exact recall: 0.000000
- Micro core relation recall: 0.000000
- Core-set completion: 0.000000
- Any-valid evidence recall: 0.000000
- Refusal accuracy: 1.000000
- Improved / regressed / unchanged questions: 0/3/7

## Safety and accounting

- Reservations settled / active / double-settled: 10/0/0
- Delivered request hashes matched: 10/10
- Citation audit pairs / inherited / pending: 0/0/0
- No retry, JSON repair, response normalization, Reranker, Gold, or human-label input.
- Historical Stage 13 results remain unchanged; Stage 13.14 remains FAILED_AND_PRESERVED.
