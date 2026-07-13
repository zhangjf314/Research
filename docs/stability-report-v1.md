# Stability Report v1

- Unique papers: 30 / target 30 (met: True)
- Parse success rate: 100.0%
- Mean ingest/parse latency: 3822.812 ms
- Mean index latency: 91.763 ms
- Retrievals: 100; mean 295.309 ms; P95 353.266 ms
- QA: mean 7.799 ms; P95 9.017 ms
- Deep Research runs: 3; latencies [1034.705, 973.202, 883.527]
- Failures: 0 (0.0%)
- Retry success: 0/0 (0.0%)
- Service restart recovered: True
- Tokens / cost: 0 / $0.00 (no real LLM)
- Total elapsed: 167.471 s

## Memory

- Conclusion: inconclusive from discrete docker stats snapshots; no soak test
- `start`: research-nginx-1=3.801MiB / 7.558GiB; research-api-1=199.3MiB / 7.558GiB; research-postgres-1=28.46MiB / 7.558GiB; research-redis-1=6.984MiB / 7.558GiB; research-qdrant-1=144.1MiB / 7.558GiB
- `after_ingestion`: research-nginx-1=4.512MiB / 7.558GiB; research-api-1=516.9MiB / 7.558GiB; research-postgres-1=28.66MiB / 7.558GiB; research-redis-1=7.91MiB / 7.558GiB; research-qdrant-1=170.4MiB / 7.558GiB
- `after_workload`: research-nginx-1=4.195MiB / 7.558GiB; research-api-1=557.2MiB / 7.558GiB; research-postgres-1=28.57MiB / 7.558GiB; research-redis-1=7.594MiB / 7.558GiB; research-qdrant-1=223.8MiB / 7.558GiB
- `after_restart`: research-nginx-1=4.184MiB / 7.558GiB; research-api-1=141.1MiB / 7.558GiB; research-postgres-1=23.96MiB / 7.558GiB; research-redis-1=5.766MiB / 7.558GiB; research-qdrant-1=290.4MiB / 7.558GiB

## Failures

- None

## Boundary

This is a bounded RC workload, not a long-duration soak test. Discrete memory snapshots can identify a large jump but cannot prove the absence of a slow leak.
