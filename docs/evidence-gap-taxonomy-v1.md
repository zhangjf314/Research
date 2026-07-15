# Evidence Gap Taxonomy v1

| Category | Retrieval issue | Auto-fix | Human review | Affects Gold |
|---|---:|---:|---:|---:|
| `candidate_recall_failure` | True | False | True | False |

**candidate_recall_failure** 鈥?Gold evidence is absent from the frozen candidate pool.
Criteria: No Gold block appears within candidate_pool_k after generic scoring.
Observable evidence: gold_candidate_rank is null or exceeds candidate_pool_k.

| `dense_rank_failure` | True | False | True | False |

**dense_rank_failure** 鈥?Dense ranking suppresses supporting evidence.
Criteria: Gold dense score/rank is materially worse than non-supporting candidates.
Observable evidence: Requires a source trace containing dense_score; unavailable in the frozen Stage 13 trace.

| `lexical_rank_failure` | True | False | True | False |

**lexical_rank_failure** 鈥?Lexical ranking suppresses supporting evidence.
Criteria: Gold lexical score/rank is materially worse despite relevant terminology.
Observable evidence: Requires a source trace containing lexical_score; unavailable in the frozen Stage 13 trace.

| `fusion_rank_failure` | True | False | True | False |

**fusion_rank_failure** 鈥?Gold evidence reaches the candidate pool but not final Top-10.
Criteria: Gold rank is within candidate_pool_k and greater than final context k.
Observable evidence: Candidate and selected ranks from the deterministic replay.

| `evidence_role_mismatch` | True | False | True | False |

**evidence_role_mismatch** 鈥?A supporting block is assigned an incompatible evidence role.
Criteria: Gold evidence roles do not intersect the routed required roles.
Observable evidence: Gold evidence_roles and router profile filters.

| `block_type_filter_error` | True | False | True | False |

**block_type_filter_error** 鈥?A supporting block is rejected by generic block/evidence eligibility.
Criteria: A Gold block exists but eligible_for_final_context is false.
Observable evidence: Block type, evidence roles, and rejection reasons.

| `section_rule_error` | True | False | True | False |

**section_rule_error** 鈥?Section compatibility rules incorrectly suppress supporting evidence.
Criteria: Gold is eligible but section scoring/caps exclude it.
Observable evidence: Section title, section score, candidate rank, packing trace.

| `context_budget_truncation` | True | False | True | False |

**context_budget_truncation** 鈥?Selected support is removed by the final token budget.
Criteria: Gold is selected before packing and appears in a truncation trace.
Observable evidence: Pre-pack selection and context packing trace.

| `page_hit_block_miss` | False | False | True | False |

**page_hit_block_miss** 鈥?The context reaches a Gold page but not an exact Gold block.
Criteria: gold_page_available=true and exact_gold_block_available=false.
Observable evidence: Selected pages and Gold pages/blocks.

| `multi_block_evidence_required` | False | False | True | True |

**multi_block_evidence_required** 鈥?A claim requires a set of blocks rather than one block.
Criteria: No individual block directly supports the complete claim.
Observable evidence: Claim text, Gold blocks, neighbors, and human source inspection.

| `gold_granularity_too_narrow` | False | False | True | True |

**gold_granularity_too_narrow** 鈥?Gold omits equivalent directly supporting evidence.
Criteria: Human review confirms a non-Gold block is equally valid support.
Observable evidence: Gold, candidate, neighbor, and source comparison.

| `gold_granularity_too_broad` | False | False | True | True |

**gold_granularity_too_broad** 鈥?A Gold block contains materially broader content than needed.
Criteria: Human review finds only a smaller span directly supports the claim.
Observable evidence: Gold block and sentence-level source inspection.

| `equivalent_non_gold_evidence` | False | False | True | True |

**equivalent_non_gold_evidence** 鈥?Selected non-Gold evidence directly supports the same claim.
Criteria: Human review confirms semantic and factual equivalence.
Observable evidence: Selected text versus Gold and source context.

| `parsing_boundary_error` | True | False | True | True |

**parsing_boundary_error** 鈥?Parsing split or merged source text across incorrect block boundaries.
Criteria: Direct support is fragmented, missing, or attached to a wrong page/block.
Observable evidence: PDF/source comparison, block neighbors, unusually short fragments.

| `claim_role_unknown` | True | False | True | False |

**claim_role_unknown** 鈥?The derived claim role is unknown and uses the safe generic route.
Criteria: claim_role=unknown for relevant ClaimUnits.
Observable evidence: ClaimUnit role and routing decision.

| `query_formulation_failure` | True | False | True | False |

**query_formulation_failure** 鈥?The retrieval query lacks terms needed to locate support.
Criteria: Human review confirms the query under-specifies the evidence need.
Observable evidence: Retrieval query, required claims, and candidate scores.

| `multi_paper_allocation_failure` | True | False | True | False |

**multi_paper_allocation_failure** 鈥?Candidate or context allocation fails to preserve all target papers.
Criteria: A multi-paper query omits at least one required paper.
Observable evidence: Target papers, selected paper distribution, per-paper ranks.

| `metric_exact_match_limitation` | False | False | True | False |

**metric_exact_match_limitation** 鈥?Strict block identity misses valid evidence at another granularity.
Criteria: Human review confirms support while exact Gold block identity is absent.
Observable evidence: Strict metric result and approved alternative evidence mapping.

| `unknown` | False | False | True | False |

**unknown** 鈥?Available automatic evidence cannot determine a cause.
Criteria: No deterministic category is justified.
Observable evidence: Incomplete or conflicting diagnostic signals.
