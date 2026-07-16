# Stage 13.12 Checkpoint File Audit

- Branch / HEAD: `eval/dev-v3-2-citation-quality-v1` / `c1e4233a2475a78ec2000f143260083143eca177`
- Raw run directories kept local-only: 10
- Uncertain files: 0
- Historical review ZIPs, raw runs, imports, backups, `.env`, secrets, and provider headers are excluded from the checkpoint.

| Category | Count |
|---|---:|
| production_code | 1 |
| evaluation_code | 6 |
| test | 1 |
| manifest | 1 |
| canonical_evaluation_data | 2 |
| canonical_report | 2 |
| provider_health | 1 |
| citation_audit | 2 |
| failure_evidence | 5 |
| raw_run_local_only | 2 |
| temporary | 0 |
| backup | 0 |
| uncertain | 0 |

| Path | Category | Commit |
|---|---|---|
| `artifacts/stage13-10-human-claim-gold-review-results.zip` | raw_run_local_only | false |
| `artifacts/stage13-9-human-citation-review-results.zip` | raw_run_local_only | false |
| `data/evaluation/evidence-qa-dev-v3-2-citation-audit-v1.jsonl` | citation_audit | true |
| `data/evaluation/evidence-qa-dev-v3-2-final-audit.json` | failure_evidence | true |
| `data/evaluation/evidence-qa-dev-v3-2-manifest.json` | manifest | true |
| `data/evaluation/evidence-qa-dev-v3-2.csv` | canonical_evaluation_data | true |
| `data/evaluation/evidence-qa-dev-v3-2.json` | canonical_evaluation_data | true |
| `data/evaluation/provider-health-dev-v3-2-v1.json` | provider_health | true |
| `data/evaluation/stage13-12-checkpoint-file-audit-v1.json` | failure_evidence | true |
| `data/evaluation/stage13-12-dev-v3-2-failure-freeze-v1.json` | failure_evidence | true |
| `docs/evidence-qa-dev-v3-2-citation-audit-v1.md` | citation_audit | true |
| `docs/evidence-qa-dev-v3-2-manifest.md` | canonical_report | true |
| `docs/evidence-qa-dev-v3-2.md` | canonical_report | true |
| `docs/stage13-12-checkpoint-file-audit-v1.md` | failure_evidence | true |
| `docs/stage13-12-dev-v3-2-failure-freeze-v1.md` | failure_evidence | true |
| `scripts/audit_evidence_qa_dev_v3_2.py` | evaluation_code | true |
| `scripts/evidence_qa_dev_v3_2_lib.py` | evaluation_code | true |
| `scripts/finalize_evidence_qa_dev_v3_2_runs.py` | evaluation_code | true |
| `scripts/freeze_stage13_12_checkpoint_v1.py` | evaluation_code | true |
| `scripts/run_evidence_qa_dev_v3_2.py` | evaluation_code | true |
| `scripts/summarize_evidence_qa_dev_v3_2.py` | evaluation_code | true |
| `src/paper_research/generation/required_claim_output.py` | production_code | true |
| `tests/test_stage13_12_dev_v3_2_live.py` | test | true |
