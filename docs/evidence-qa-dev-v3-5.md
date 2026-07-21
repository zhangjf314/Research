# Evidence QA Dev v3.5 Controlled Live

## Shape reliability

- Provider completed: 10/10
- JSON valid / Payload v4 schema / Slot shape success: 10/10, 9/10, 9/10
- Status / citation / null / empty leakage: 0/0/0/0
- Slot shapes answered / unsupported / invalid: 24/0/3 of 27

## Quality metrics

- Required claim macro exact recall: 0.166667
- Citation recall micro core relation: 0.200000
- Any-valid evidence recall: 0.259259
- Core-set completion: 0.148148
- Refusal accuracy: 1.000000
- Improved / regressed / unchanged questions: 2/1/7

## Safety

- Requests / tokens / cost: 10, 19877, USD 0
- Active reservations / retries / reranker / template fallback: 0/0/False/False
- No normalization, JSON repair, retry, citation repair, fallback, Gold injection, or human-label injection.
