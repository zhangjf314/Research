# Portfolio QA Limitations v1

- The evaluation uses a 50-item human-reviewed internal development set, not an independent blind benchmark.
- The system deterministically validates citation IDs, context membership, and block/page mappings.
- Complete semantic entailment between every claim and citation has not been formally validated at scale.
- Gold citation exact-match metrics may underestimate valid citations that use equivalent blocks or pages.
- Do not claim production-grade grounding, strict blind-test generalization, or fully eliminated hallucination.
