# Claim Gold review guide v1

Review every required claim independently. Candidate provenance and automated/human citation
signals are context, not Gold labels.

Use `core_gold` for a relation that alone supports the claim, or place all relations needed for a
minimum complete multi-block set in one core set. Use `supporting_gold` only when the block belongs
to a complete evidence set but is not sufficient alone. Use `equivalent_valid_evidence` for valid
evidence outside historical question-level Gold; it does not rewrite historical exact Gold.
`partially_relevant`, `insufficient`, and `unrelated` are not formal Gold. If the corpus contains no
valid evidence, select `no_valid_evidence`; do not force a block.

Pay special attention to q001 decomposition, q004's GPU/Adam/warmup set, q015 paper identity versus
future work, q019 numeric and experimental subclaims, and q050 comparison-side completeness.

Example:

```
python scripts/review_claim_evidence_gold_dev_v1.py --required-claim-id <id>   --approve-core <relation-id> --reviewer <name> --notes "<reason>"
```

Repeat `--approve-core` to create one multi-relation minimum complete core set. The tool always
backs up the JSONL before a write and never approves a record without an explicit human action.
