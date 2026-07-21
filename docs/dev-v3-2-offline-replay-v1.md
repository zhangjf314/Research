# Dev v3.2 offline citation replay v1

- Questions / slots: 10 / 27
- Provider / Embedding calls: 0 / 0
- Replay limitation: preflight only; it cannot predict new-prompt generation behavior.

| Metric | Baseline | Full candidate |
|---|---:|---:|
| Average citations | 1.222222 | 1.000000 |
| Obligation complete | 0.666667 | 0.962963 |
| Numeric complete | 0.925926 | 1.000000 |
| Comparison complete | 0.962963 | 1.000000 |
| Exact relation recall | 0.185185 | 0.166667 |
| Any-valid recall | 0.296296 | 0.333333 |
| Core-set completion | 0.185185 | 0.148148 |
| Wrong evidence | 4 | 1 |
| Citation dilution | 0.151515 | 0.000000 |

Human support proxies are **offline label-based diagnostics only** and are not selection features.
