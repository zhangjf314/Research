# Stage 13 Baseline Freeze v1

> Offline deterministic freeze. Gold is joined only after selection. No LLM, Embedding, Reranker, or Deep Research call was made.

- Code commit: `09fe3a17e1bbb6fd1694b55fe11a1618d0eae864`
- Corpus signature: `fb826cb8b1fe8e55d96b7aa1446cfb2e8d347efd74643a627706ad00e3792239`
- Configuration fingerprint: `c15f2c22e44771daa31ee5837cb672cbba643aa9558af80afc9c514d016bd180`
- Exact block availability: **31/48 = 0.645833**
- Gold page availability: **42/48 = 0.875000**
- Gold block recall: **0.319444**
- Multi-paper coverage: **1.000000**
- Metadata contamination: **0.000000**
- Mean context tokens: **1697.50**
- P95 latency: **23.624800 ms**

The JSON freeze contains every input SHA-256 and each question's selected Evidence IDs. The replay implementation constructs the selection from the approved retrieval protocol and frozen source context before reading Gold fields for evaluation.
