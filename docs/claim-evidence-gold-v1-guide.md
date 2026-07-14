# Claim Evidence Gold v1 Review Guide

This file is a pending draft. Automation must never change `annotation_status` to `approved`.

For each claim, record one or more evidence sets. A set is sufficient as a whole: one triple means
one block is sufficient; several triples mean the blocks are jointly required. Equivalent sets
belong in `acceptable_alternative_evidence`. Page proximity alone is insufficient. For an
unanswerable obligation, leave supporting sets empty and document the reviewed negative evidence.

Every triple must match the immutable `(paper_id, page, block_id)` source. Reviewer and notes are
mandatory. The CLI creates a timestamped backup and refuses to overwrite an approved row unless
`--overwrite` is explicit. If the claim fingerprint or source evidence version changes, invalidate
the old approval and review again.

Example syntax:

```powershell
.\.venv\Scripts\python.exe scripts\review_claim_evidence_gold_v1.py `
  --claim-id <claim-id> --reviewer <name> --notes "reason" `
  --evidence-set "1706.03762:2:b000022,1706.03762:2:b000025" --approve
```

Use `--list-pending` to list outstanding claims. Question/category/difficulty arguments are
reserved for filtered review batches; they never modify labels automatically.
