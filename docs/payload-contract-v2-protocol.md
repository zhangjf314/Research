# Payload Contract v2 Protocol

- Selected option: A — answerable accepts null or exact empty string.
- Canonicalization: exact `"" -> null` at `$.refusal_reason` only.
- Final envelope always uses null for answerable responses.
- This is versioned non-semantic canonicalization, not normalization, JSON repair, semantic repair, schema bypass, or fuzzy validation.
- `NEXT_LIVE_AUTHORIZED=false`.
