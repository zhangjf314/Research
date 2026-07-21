# Dev v3.1 Citation Failure Taxonomy

- `adjacent_partial_support`: 1 claims; questions=['q007']; generic fix=treat adjacent completion as supporting rather than automatically primary evidence
- `citation_fully_supported`: 12 claims; questions=['q001', 'q002', 'q004', 'q007', 'q008', 'q013', 'q050']; generic fix=retain strict claim-local validation
- `citation_partially_supported`: 5 claims; questions=['q008', 'q013', 'q019']; generic fix=decompose compound claims and verify numeric/comparative completeness
- `citation_unsupported`: 2 claims; questions=['q015']; generic fix=require claim-local lexical/numeric/comparative evidence checks
- `equivalent_non_gold_cited`: 3 claims; questions=['q004', 'q015', 'q019']; generic fix=preserve semantic support separately from exact-Gold diagnostics
- `same_page_boundary_miss`: 4 claims; questions=['q001', 'q007', 'q050']; generic fix=score claim-local completeness and prefer the primary block before adjacent support

No question/block special case, Gold injection, or human-label production selection is used.
