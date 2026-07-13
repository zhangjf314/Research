# Soak Test v1

- Duration: 182.703 seconds
- Queries: 276; mean 268.794 ms; P95 333.428 ms
- Periodic imports: 4
- Deep Research runs: 3
- API restart performed: True
- Failures: 0
- Token / cost: 0 / $0.00 (baseline template provider; not production-model evidence)

## Resource samples

| Seconds | API memory | PostgreSQL state | Qdrant points | Redis keys |
|---:|---:|---|---:|---:|
| 0.125 | 397.8MiB / 7.558GiB | 2|9|50 | 2065 | 5 |
| 15.297 | 434.8MiB / 7.558GiB | 2|18|100 | 2065 | 6 |
| 30.453 | 439.1MiB / 7.558GiB | 2|18|100 | 2065 | 7 |
| 45.359 | 439.4MiB / 7.558GiB | 2|18|100 | 2065 | 6 |
| 60.078 | 439.6MiB / 7.558GiB | 2|18|100 | 2065 | 6 |
| 75.156 | 450.3MiB / 7.558GiB | 2|27|150 | 2065 | 6 |
| 90.156 | 457MiB / 7.558GiB | 2|27|150 | 2065 | 7 |
| 105.156 | 203.5MiB / 7.558GiB | 2|27|150 | 2065 | 7 |
| 120.453 | 199.5MiB / 7.558GiB | 2|27|150 | 2065 | 6 |
| 135.453 | 208.2MiB / 7.558GiB | 2|36|200 | 2065 | 6 |
| 150.172 | 202.9MiB / 7.558GiB | 2|36|200 | 2065 | 5 |
| 165.187 | 205.1MiB / 7.558GiB | 2|36|200 | 2065 | 6 |
| 180.141 | 209.9MiB / 7.558GiB | 2|36|200 | 2065 | 6 |

## Interpretation boundary

This is a time-bounded local soak. It can expose immediate restart, connection, and memory-growth defects, but cannot prove the absence of slow leaks over production-scale durations.
