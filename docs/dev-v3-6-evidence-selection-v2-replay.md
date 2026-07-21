# Dev v3.6 Evidence Selection v2 Replay

- Selection version: `evidence-selection-v2-candidate`
- Replay hash: `3c281220fbc279731449557cee7c5d465ffd1c2cf507edee589c15e51ff3a286`
- Offline quality preflight: `FAILED`
- Retrieval completion required: `False`
- Gold and human labels are used only by offline scorers, never by selection.

| Mode | Any-valid | Claim macro exact | Wrong evidence |
|---|---:|---:|---:|
| baseline | 0.259259 | 0.185185 | 16 |
| selection_v2_only | 0.296296 | 0.259259 | 15 |
| full_candidate | 0.407407 | 0.259259 | 0 |
