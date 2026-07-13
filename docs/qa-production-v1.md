# Production QA Evaluation v1

- Mode: full
- Model: `Qwen/Qwen3-8B` via SiliconFlow
- Prompt: `qa-production-v1`
- Retrieval: Jina 1024d + Structural Hybrid, Recall 20 / Top 10
- Reranker: disabled
- Deep Research: not run
- Completed/failures/retries: 50/0/2

## Metrics

- JSON parse success: 1.0
- Schema validation success: 1.0
- Answerable accuracy: 0.875
- Refusal accuracy: 1.0
- Required claim coverage: 0.388889
- Unsupported claims: 147
- Citation presence / validity: 1.0 / 1.0
- Citation precision / recall: 0.103009 / 0.096875
- Claim-citation binding: 1.0
- Gold block retrieved: 0.4375
- Total latency mean/p50/p95 ms: {'mean_ms': 18701.054, 'p50_ms': 14911.625, 'p95_ms': 42170.931}
- First-token latency: {'mean_ms': None, 'p50_ms': None, 'p95_ms': None}
- Tokens input/output/total: 307953/20753/328706
- Estimated cost USD: None (configured=False)
- API requests / rate limits: 52 / 0

Rule-based required-claim matching uses token-set recall with threshold 0.35 and stores every raw score. No LLM judge is used.

## Smoke / Dev / Full progression

| Mode | Completed | Failures | Retries | Answerable | Refusal | Claim coverage | Citation precision | Citation recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| smoke | 3 | 0 | 0 | 1.0 | 1.0 | 0.333333 | 0.0 | 0.0 |
| dev | 10 | 0 | 1 | 1.0 | 1.0 | 0.5 | 0.160714 | 0.133333 |
| full | 50 | 0 | 2 | 0.875 | 1.0 | 0.388889 | 0.103009 | 0.096875 |

## Final retry and failure records

- `q003`: retries=1, reasons=['citation_validation']
- `q049`: retries=1, reasons=['malformed_json']