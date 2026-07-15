# Evidence QA Dev Citation Review Guide v1

Review all 24 claim-citation pairs against the cited block and adjacent context. Automated exact/page/semantic fields are routing aids, not human labels.

Valid labels: `fully_supported`, `partially_supported`, `related_but_insufficient`, `unsupported`, `gold_annotation_too_narrow`, `ambiguous_claim`, `malformed_evidence`.

Example:

```powershell
.\.venv\Scripts\python.exe scripts\review_evidence_qa_dev_citations_v1.py --sample-id evidence-qa-dev-citation-001 --label fully_supported --reviewer <name> --notes "Directly supported by the cited block."
```

Reviewer and notes are mandatory. Existing reviews require `--overwrite`. Every update creates a backup and fails if source hashes or citation triples changed. No model or automated signal may fill the human label.
