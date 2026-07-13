# OCR End-to-End Audit v1

## Docker revalidation — Stage 10 (2026-07-13)

The previous container limitation is superseded: the rebuilt image installs Tesseract
5.5.0 and English language data. Actual Docker API uploads produced:

| Type | Parser | OCR | Source pages | Chunk | QA/claim citation |
|---|---|---:|---|---:|---|
| Text | `pymupdf` | false | 1 | 1 | passed, page 1 |
| Mixed | `ocr` | true | 1–2 | 1 | passed, pages 1–2 |
| Fully scanned | `ocr` | true | 1 | 1 | passed, page 1 |

`ocr_confidence` remains `null` with `OCR_CONFIDENCE_UNAVAILABLE`, because the PyMuPDF
OCR text-page integration does not expose Tesseract word confidence. This is an explicit
metadata limitation, not a silent success value. Docling and GROBID remain optional and
unverified.

- Tesseract: `D:\Program Files\Tesseract-OCR\tesseract.exe`
- TESSDATA_PREFIX: `D:\Program Files\Tesseract-OCR\tessdata`
- OCR is an optional fallback, not the default text-PDF path.
- OCR confidence is recorded as `null`: PyMuPDF's OCR text-page API does not expose word confidence.

| PDF type | Parser | OCR | Blocks | Chunks | Text recovered | Order | Page | Indexed | QA | Citation range | Contains evidence page | Parse ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| text | pymupdf | False | 2 | 1 | True | True | True | True | True | 1-1 | True | 13.537 |
| mixed | ocr | True | 3 | 1 | True | True | True | True | True | 1-2 | True | 533.096 |
| scanned | ocr | True | 2 | 1 | True | True | True | True | True | 1-1 | True | 367.224 |

## Installation boundary

Host OCR tests require `D:\Program Files\Tesseract-OCR\tesseract.exe` and `D:\Program Files\Tesseract-OCR\tessdata\eng.traineddata`. The current API Docker image does not install Tesseract, so OCR inside the deployed container remains unverified/unsupported until the image adds the package and language data.
