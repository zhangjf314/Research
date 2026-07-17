# Dev v3.6 Evidence Selection v4 Replay

- Selection version: `evidence-selection-v4-candidate`
- Replay hash: `90021ba0b3c403b51408a0c9d7b0bf134ff372349486d50f7bc91e5c8ce46471`
- Engineering Gate: `PASSED`
- Quality Preflight: `FAILED`
- Next live ready: `False`

## Full V4 candidate

- Any-valid recall: `0.2962962962962963`
- Question macro exact: `0.2222222222222222`
- Claim macro exact: `0.2222222222222222`
- Micro core: `0.2222222222222222`
- Core-set completion: `0.18518518518518517`
- Aligned wrong evidence: `15`
- Avg citations: `1.173913043478261`
- Improved/regressed/unchanged: `1` / `0` / `23`
- q002 regressed: `False`

## Modes

- `A_stage13_21_baseline`: any-valid=`0.25925925925925924`, wrong=`16`
- `B_selection_v2`: any-valid=`0.2962962962962963`, wrong=`15`
- `C_selection_v3_protected`: any-valid=`0.2222222222222222`, wrong=`16`
- `D_v4_baseline_validation_only`: any-valid=`0.25925925925925924`, wrong=`16`
- `E_v4_add_complement_only`: any-valid=`0.2962962962962963`, wrong=`15`
- `F_v4_replacement_only`: any-valid=`0.25925925925925924`, wrong=`16`
- `G_v4_baseline_first_combined`: any-valid=`0.2962962962962963`, wrong=`15`
- `H_v4_candidate_admission_v3`: any-valid=`0.2962962962962963`, wrong=`15`
- `I_v4_fallback_v3`: any-valid=`0.2962962962962963`, wrong=`15`
- `J_full_v4_candidate`: any-valid=`0.2962962962962963`, wrong=`15`
- `K_full_v4_without_baseline_protection`: any-valid=`0.2962962962962963`, wrong=`15`
- `L_full_v4_with_old_candidate_veto`: any-valid=`0.2962962962962963`, wrong=`15`
- `M_oracle_candidate_upper_bound`: any-valid=`0.4074074074074074`, wrong=`0`
