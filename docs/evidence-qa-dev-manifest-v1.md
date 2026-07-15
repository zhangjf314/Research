# Evidence-Centric Dev Manifest v1

- Manifest hash: `fcb59b71fc68549479c24f6475f7d18ad9e382aace93e70e93594ee355ffb988`
- Frozen before Stage 13.2 live results: **yes**
- Result-dependent resampling: **no**
- Historical baseline source: `historical_stage11c`
- Evidence-centric variant C: `blocked_by_known_selector_defect`

| ID | Category | Difficulty | Scope | Answerable | Baseline exact | Phase B exact | Inclusion |
|---|---|---|---|---:|---:|---:|---|
| q001 | research_background | medium | paper | True | True | True | baseline_already_hit_control |
| q002 | paper_contributions | easy | paper | True | False | True | phase_b_exact_hit_gain |
| q004 | experiment_setup | hard | paper | True | True | True | baseline_already_hit_control |
| q005 | unanswerable | easy | unanswerable | False | None | None | unanswerable_refusal |
| q007 | paper_contributions | hard | paper | True | False | True | phase_b_exact_hit_gain |
| q008 | algorithm_steps | easy | paper | True | True | True | baseline_already_hit_control |
| q013 | method | easy | paper | True | False | True | phase_b_exact_hit_gain |
| q015 | limitations | hard | paper | True | False | False | persistent_retrieval_miss |
| q019 | experiment_results | hard | paper | True | False | False | persistent_retrieval_miss |
| q050 | multi_paper_comparison | hard | multi_paper | True | False | True | phase_b_exact_hit_gain, multi_paper_coverage |
