# Citation Human Audit Guide v1

This file contains 30 pending claim-citation judgments: 10 semantic non-Gold,
10 same-Gold-page non-exact, and 10 unsupported/weak automated cases.

Review the claim against the cited block text and the Gold block text. Do not infer support
from the paper title or outside knowledge. `human_label` must be one of:

- `fully_supported`
- `partially_supported`
- `related_but_insufficient`
- `unsupported`
- `gold_annotation_too_narrow`

Only a human reviewer may replace `human_review_status=pending`, populate `human_label`,
and add `review_notes`. `suggested_human_label` is an automated routing hint, not a human
conclusion, and must not be copied into `human_label` without review.
