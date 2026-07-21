# Evidence Selection v3 Recall Collapse Audit

- Attribution: `COMPLETE`
- Primary failure: `true_no_valid_candidate`
- Candidate upper-bound hit claims: `11`
- V3 final hit claims: `6`
- Recall lost claims: `5`
- Unknown reasons: `0`

## Root cause distribution

- `baseline_removed_without_gain`: `1`
- `claim_fallback_regression`: `1`
- `complementary_candidate_rejected`: `4`
- `no_failure`: `6`
- `true_no_valid_candidate`: `15`

## Baseline replacements

- `q002` `cl-q002-8a1d729edcaafb379a20`: ['E005'] -> ['E001']; classification=`baseline_removed_without_gain`
- `q013` `cl-q013-938d2a4e18520c28a5d7`: ['E005'] -> []; classification=`claim_fallback_regression`
