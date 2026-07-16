# Evidence QA Dev v3.1 Human Citation Audit Summary

- Audit nature: **AI-assisted manual citation audit**
- Reviewed: 33/33
- Labels: fully=16, partial=10, related=5, unsupported=2
- Strict/lenient support: 0.484848/0.787879
- Exact miss but lenient-supported: 18/24 = 0.750000
- Exact hit but not fully supported: 3/9 = 0.333333
- The external narrative stated 17/24 exact misses were supported; deterministic recomputation from the reviewed labels gives 18/24. Labels were not changed.
- Exact Gold hit does not guarantee full support; Exact Gold miss does not imply an invalid citation.
- Human support and Exact Gold Recall remain separate metrics.
- This fixed 10-question, 33-pair AI-assisted audit cannot be extrapolated to Full-50 or production.
