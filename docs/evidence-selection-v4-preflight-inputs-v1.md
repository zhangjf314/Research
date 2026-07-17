# Evidence Selection v4 Preflight Inputs

- Signature: `802e12b4905525e3a88ef22a6912da3d1f9c638248f572f51267adc8d4596ee8`
- Candidate budget: `12`
- Citation budget: `{'max_primary': 1, 'max_supporting': 2, 'max_total': 3}`
- Gold freeze is recorded for offline scoring only.
- No live LLM, embedding API, reranker, Human Citation Audit, Full QA, or Deep Research was executed by this freeze step.

## Frozen file hashes

- `stage13_21_summary`: `f944dffa6ff5ee7c8dcc5662a7fb82344e021c0579283a6c812c7d4c49752726` (`data\evaluation\evidence-qa-dev-v3-6.json`)
- `stage13_21_final_audit`: `040fa74bf0aca309da4201b2beb46fa5639850876ff76aa77909db53bb1d3a29` (`data\evaluation\evidence-qa-dev-v3-6-final-audit.json`)
- `stage13_21_citation_traces`: `2fc856c44fd1c0bdf7626c55e06a41effc5420d5df7e479eca46c2f128aff60c` (`data\evaluation\evidence-qa-dev-v3-6-citation-audit-v1.jsonl`)
- `stage13_22_evidence_funnel`: `49609c63634795a89871d557eb747f45bef9593baf1b8cbcec1cc14b81762223` (`data\evaluation\dev-v3-6-evidence-funnel-v1.jsonl`)
- `stage13_22_attribution`: `02538139b8b4b2a3450ff3884717193ea6ef806a11889998bde968166966ec11` (`data\evaluation\dev-v3-6-quality-failure-attribution-v1.json`)
- `stage13_23_wrong_evidence_audit`: `d1d6a274489c46c51b73ebf7e47172473331a7979e5dd491f4dc7d498d132460` (`data\evaluation\evidence-selection-v2-wrong-evidence-audit-v1.json`)
- `wrong_evidence_metric_alignment`: `34047e0357baaf58654f1de6b5714f37d162bc2878b6f2aa002f9428cfc89a01` (`data\evaluation\wrong-evidence-metric-alignment-v1.json`)
- `selection_v2_replay`: `d7bdbbe8da13ac49155835605021fe7c514c731fde9812422532c85e29f5bb9e` (`data\evaluation\dev-v3-6-evidence-selection-v2-replay.json`)
- `selection_v3_replay`: `28832e91ca00f3da0bad8dabc2a82949e5e51985d5d9dabfe5c1672336b7a1a3` (`data\evaluation\dev-v3-6-evidence-selection-v3-replay.json`)
- `selection_v3_readiness`: `a179f729a5cc6b209f5af791ad1147b05b02e0522657a33e739a891fc9f61a9a` (`data\evaluation\stage13-23-selection-v3-readiness-v1.json`)
- `claim_gold_freeze`: `604d359526f61ca46c6a9a399f4b21f3eb945a4f992caa859ed7d4a1d6522bf8` (`data\evaluation\claim-evidence-gold-dev-v1-freeze.json`)
- `claim_gold`: `e1aadcae82fb8a7f867eb600ffb0b2836813fd70e38c70502992cc7e4faa4bcd` (`data\evaluation\claim-evidence-gold-dev-v1.jsonl`)
- `payload_v4_protocol`: `02d80f90da7a4d7ecc2d1917b769b5b72c459ff0009caabda776ce3cce6559d0` (`data\evaluation\payload-contract-v4-protocol.json`)
- `payload_v4_preflight`: `f6feca6c166fb9b089552702962f6b99d5792452167e49f9a76388746bbcaba1` (`data\evaluation\payload-contract-v4-preflight-inputs-v1.json`)
- `envelope_v4`: `3a7662112951c699726ac93b471e747b02cbedadd740b1cfafee01c610359e8b` (`data\evaluation\dev-v3-4-payload-contract-v4-final-audit.json`)
- `evidence_presentation_v2`: `ef9d5213609f99b4bbba742f6f7b09c61b07f6d05e785a79a5577c6cbaf73f02` (`data\evaluation\evidence-presentation-v2-protocol.json`)
- `prompt_v3_7`: `31d8dc3dc1aa4435df62f2dc310be5107b780ef1e0b9fae9fb11d2e2297619e1` (`data\evaluation\dev-v3-6-prompt-rendering-preflight-v1.json`)
- `citation_budget`: `f19d0aa70c54f1d4c4ea1814d8a1807ab57315f43ce7706055abf00caa70abd3` (`src\paper_research\generation\citation_selection.py`)
