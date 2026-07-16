# Schema Reliability v1 Candidate

- Selected: **A+B** — minimal model payload plus locally bound immutable envelope, with citation selection owned by deterministic local policy.
- The model no longer copies prompt/citation protocol constants and does not emit citation IDs.
- Malformed JSON remains a strict failure; no repair or normalization is added.
- This is a new offline candidate, not a Dev v3.2 rerun or Dev v3.3 authorization.
- `NEXT_LIVE_AUTHORIZED=false`.
