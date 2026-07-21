# Docker OCR Production Audit v2

- Gate: `PASSED`
- Docker tesseract path: `/usr/bin/tesseract`
- Text PDF roundtrip: `passed`
- Mixed PDF roundtrip: `passed`
- Scanned PDF roundtrip: `passed`
- Citation page accuracy: `1.0`

| PDF type | Upload | Indexed | OCR blocks | Text recovered | Retrievable | Citation page | Expected page | Page accuracy | Error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| TEXT_PDF | passed | True | 0 | True | True | 1-1 | 1 | True | None |
| MIXED_PDF | passed | True | 4 | True | True | 1-2 | 2 | True | None |
| SCANNED_PDF | passed | True | 2 | True | True | 1-1 | 1 | True | None |

Synthetic PDFs are non-sensitive local fixtures retained under `artifacts/` for audit. This script exercises the Docker API upload, parse, index, retrieval, and citation-page mapping path; it does not run the formal 50-item QA evaluation.
