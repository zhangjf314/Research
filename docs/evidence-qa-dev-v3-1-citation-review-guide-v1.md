# Evidence QA Dev v3.1 Citation Review Guide

Review all 33 rows independently. Do not infer a label from exact/page/semantic automated signals. Verify whether the cited evidence supports the generated claim, using previous/current/next context and the supplied Gold only as reference.

Allowed labels:

- `fully_supported`
- `partially_supported`
- `related_but_insufficient`
- `unsupported`
- `gold_annotation_too_narrow`
- `ambiguous_claim`
- `malformed_evidence`

For every approved row fill `human_label`, `reviewer`, `reviewed_at`, and a non-empty `review_notes`, then set `human_review_status=approved`. Do not edit sample IDs, claims, citations, evidence, Gold, automated labels, source hashes, registry hashes, or immutable hashes.
