# Dev v3.6 Quality Failure Attribution

- Attribution: `COMPLETE`
- Primary bottleneck: `MIXED`
- Evidence Selection v2 Engineering Gate: `PASSED`
- Evidence Selection v2 Quality Preflight: `FAILED`
- Retrieval completion required: `False`
- Next live ready: `False`
- Next live authorized: `False`
- Human citation review deferred: `True`

## Funnel counts

- F2 retrieval misses: `14`
- F3 candidate pruning misses: `2`
- F5 selection misses: `4`
- F6 selected-not-cited: `0`
- F7 support completeness failures: `2`
- Narrowing losses: `1`
- Unsupported losses: `0`

## Selection v2 candidate

- Any-valid recall: `0.2962962962962963`
- Question macro exact: `0.25925925925925924`
- Claim macro exact: `0.25925925925925924`
- Micro core relation: `0.25925925925925924`
- Core-set completion: `0.2222222222222222`
- Wrong evidence: `15`
- Improvement/regression/unchanged: `2` / `3` / `22`

Selection v2 remains an offline candidate. Because quality preflight failed, no new Dev live run, Human Citation Audit, Full QA, or Deep Research is authorized.
