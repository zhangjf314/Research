from __future__ import annotations

# ruff: noqa: E501
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fitz
import httpx

API_BASE_URL = "http://localhost/api/v1"
ARTIFACT_DIR = Path("artifacts/docker-ocr-production-v2")
OUTPUT_JSON = Path("data/evaluation/docker-ocr-production-v2.json")
OUTPUT_MD = Path("docs/docker-ocr-production-audit-v2.md")


def run(*args: str) -> str:
    completed = subprocess.run(args, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return completed.stdout.strip()


def docker(*args: str) -> str:
    return run("docker", "compose", *args)


def make_text_pdf(path: Path, title: str, sentence: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 100), title, fontsize=18)
    page.insert_text((72, 150), sentence, fontsize=11)
    document.save(path)
    document.close()


def image_page(title: str, sentence: str) -> bytes:
    source = fitz.open()
    page = source.new_page()
    page.insert_text((72, 100), title, fontsize=18)
    page.insert_text((72, 150), sentence, fontsize=12)
    payload = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False).tobytes("png")
    source.close()
    return payload


def make_scanned_pdf(path: Path, title: str, sentence: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_image(page.rect, stream=image_page(title, sentence))
    document.save(path)
    document.close()


def make_mixed_pdf(path: Path, title: str, text_sentence: str, scanned_sentence: str) -> None:
    document = fitz.open()
    first = document.new_page()
    first.insert_text((72, 100), title, fontsize=18)
    first.insert_text((72, 150), text_sentence, fontsize=11)
    second = document.new_page()
    second.insert_image(second.rect, stream=image_page("Mixed scanned page", scanned_sentence))
    document.save(path)
    document.close()


def container_parse_summary(paper_id: str, expected_sentence: str) -> dict[str, Any]:
    code = r"""
import json
import sys
from pathlib import Path

paper_id = sys.argv[1]
expected = sys.argv[2].lower()
root = Path('/app/data/parsed') / paper_id
blocks_path = root / 'paper_blocks.jsonl'
chunks_path = next(iter(sorted(root.glob('paper_chunks.*.jsonl'))), root / 'paper_chunks.jsonl')
blocks = []
chunks = []
if blocks_path.exists():
    blocks = [json.loads(line) for line in blocks_path.read_text(encoding='utf-8').splitlines() if line.strip()]
if chunks_path.exists():
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding='utf-8').splitlines() if line.strip()]
combined = ' '.join(str(block.get('text', '')) for block in blocks).lower()
print(json.dumps({
    'blocks_path_exists': blocks_path.exists(),
    'chunks_path_exists': chunks_path.exists(),
    'block_count': len(blocks),
    'chunk_count': len(chunks),
    'parser_names': sorted({str(block.get('parser_name', '')) for block in blocks if block.get('parser_name')}),
    'ocr_block_count': sum(1 for block in blocks if block.get('is_ocr')),
    'source_pages': sorted({block.get('source_page') or block.get('page_start') for block in blocks}),
    'text_recovered': ' '.join(expected.split()) in ' '.join(combined.split()),
    'reading_order_preserved': all(block.get('block_index') == index for index, block in enumerate(blocks)),
    'normalized_blocks': all(block.get('block_id') and block.get('text') and block.get('bbox') for block in blocks),
}))
"""
    raw = docker("exec", "-T", "api", "python", "-c", code, paper_id, expected_sentence)
    return json.loads(raw)


def post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = httpx.post(f"{API_BASE_URL}{path}", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def upload_pdf(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        response = httpx.post(
            f"{API_BASE_URL}/papers/upload",
            files={"file": (path.name, handle, "application/pdf")},
            timeout=120,
        )
    response.raise_for_status()
    return response.json()


def index_paper(paper_id: str) -> dict[str, Any]:
    response = httpx.post(f"{API_BASE_URL}/papers/{paper_id}/index", timeout=240)
    response.raise_for_status()
    return response.json()


def retrieve(paper_id: str, query: str) -> dict[str, Any]:
    return post_json(
        "/retrieve",
        {
            "query": query,
            "filters": {"paper_ids": [paper_id]},
            "recall_k": 20,
            "top_k": 5,
        },
    )


def evaluate_case(kind: str, path: Path, expected_sentence: str, expected_page: int) -> dict[str, Any]:
    started = time.perf_counter()
    row: dict[str, Any] = {
        "pdf_type": kind,
        "path": str(path),
        "expected_page": expected_page,
    }
    try:
        upload = upload_pdf(path)
        paper_id = upload["paper"]["id"]
        row["paper_id"] = paper_id
        row["upload"] = "passed"
        row["duplicate"] = upload.get("duplicate")
        row["index_response"] = index_paper(paper_id)
        row["indexed"] = row["index_response"].get("status") == "READY"
        parse = container_parse_summary(paper_id, expected_sentence)
        row.update(parse)
        retrieval = retrieve(paper_id, expected_sentence)
        context = retrieval.get("context", [])
        row["retrievable"] = bool(context)
        row["citation_page"] = context[0].get("page_start") if context else None
        row["citation_page_end"] = context[0].get("page_end") if context else None
        row["citation_page_accuracy"] = bool(context) and (
            context[0].get("page_start") <= expected_page <= context[0].get("page_end")
        )
        row["retrieval_trace_id"] = retrieval.get("trace", {}).get("trace_id")
        row["error"] = None
    except Exception as exc:  # audit script must retain per-case failures
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["indexed"] = False
        row["retrievable"] = False
        row["citation_page_accuracy"] = False
    row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return row


def main() -> None:
    started = datetime.now(UTC)
    stamp = started.strftime("%Y%m%d%H%M%S")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    sentences = {
        "text": f"Stage 13 40 text PDF evidence {stamp} says docker parsing preserves page one.",
        "mixed_text": f"Stage 13 40 mixed native evidence {stamp} appears on page one.",
        "mixed_scan": f"Stage 13 40 mixed scanned evidence {stamp} appears on page two.",
        "scanned": f"Stage 13 40 scanned PDF evidence {stamp} requires docker OCR.",
    }
    files = {
        "text": ARTIFACT_DIR / f"text-native-{stamp}.pdf",
        "mixed": ARTIFACT_DIR / f"mixed-native-scanned-{stamp}.pdf",
        "scanned": ARTIFACT_DIR / f"fully-scanned-{stamp}.pdf",
    }
    make_text_pdf(files["text"], "Docker OCR text PDF", sentences["text"])
    make_mixed_pdf(
        files["mixed"],
        "Docker OCR mixed PDF",
        sentences["mixed_text"],
        sentences["mixed_scan"],
    )
    make_scanned_pdf(files["scanned"], "Docker OCR scanned PDF", sentences["scanned"])

    tesseract_path = docker(
        "exec",
        "-T",
        "api",
        "python",
        "-c",
        "import shutil; print(shutil.which('tesseract') or '')",
    )
    cases = [
        evaluate_case("TEXT_PDF", files["text"], sentences["text"], 1),
        evaluate_case("MIXED_PDF", files["mixed"], sentences["mixed_scan"], 2),
        evaluate_case("SCANNED_PDF", files["scanned"], sentences["scanned"], 1),
    ]
    text_passed = (
        cases[0]["error"] is None
        and cases[0].get("text_recovered")
        and cases[0].get("indexed")
        and cases[0].get("retrievable")
        and cases[0].get("citation_page_accuracy")
    )
    mixed_passed = (
        cases[1]["error"] is None
        and cases[1].get("text_recovered")
        and cases[1].get("indexed")
        and cases[1].get("retrievable")
        and cases[1].get("citation_page_accuracy")
        and cases[1].get("ocr_block_count", 0) > 0
    )
    scanned_passed = (
        cases[2]["error"] is None
        and cases[2].get("text_recovered")
        and cases[2].get("indexed")
        and cases[2].get("retrievable")
        and cases[2].get("citation_page_accuracy")
        and cases[2].get("ocr_block_count", 0) > 0
    )
    gate = (
        "PASSED"
        if text_passed
        and mixed_passed
        and scanned_passed
        and bool(tesseract_path)
        else "FAILED"
    )
    payload = {
        "schema_version": "docker-ocr-production-v2",
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "gate": gate,
        "api_base_url": API_BASE_URL,
        "docker_tesseract_path": tesseract_path,
        "docker_tesseract_called_for_scanned": cases[2].get("ocr_block_count", 0) > 0,
        "text_pdf_roundtrip": "passed" if text_passed else "failed",
        "mixed_pdf_roundtrip": "passed" if mixed_passed else "failed",
        "scanned_pdf_roundtrip": "passed" if scanned_passed else "failed",
        "indexed": all(case.get("indexed") for case in cases),
        "retrievable": all(case.get("retrievable") for case in cases),
        "citation_page_accuracy": (
            1.0
            if all(case.get("citation_page_accuracy") for case in cases)
            else sum(1 for case in cases if case.get("citation_page_accuracy")) / len(cases)
        ),
        "temporary_files_cleaned": False,
        "temporary_files_policy": "synthetic fixture PDFs retained for audit under artifacts/",
        "cases": cases,
    }
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(payload)
    print(json.dumps({"gate": gate, "cases": len(cases)}, indent=2))


def write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# Docker OCR Production Audit v2",
        "",
        f"- Gate: `{payload['gate']}`",
        f"- Docker tesseract path: `{payload['docker_tesseract_path']}`",
        f"- Text PDF roundtrip: `{payload['text_pdf_roundtrip']}`",
        f"- Mixed PDF roundtrip: `{payload['mixed_pdf_roundtrip']}`",
        f"- Scanned PDF roundtrip: `{payload['scanned_pdf_roundtrip']}`",
        f"- Citation page accuracy: `{payload['citation_page_accuracy']}`",
        "",
        "| PDF type | Upload | Indexed | OCR blocks | Text recovered | Retrievable | Citation page | Expected page | Page accuracy | Error |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for case in payload["cases"]:
        lines.append(
            f"| {case['pdf_type']} | {case.get('upload')} | {case.get('indexed')} | "
            f"{case.get('ocr_block_count')} | {case.get('text_recovered')} | "
            f"{case.get('retrievable')} | {case.get('citation_page')}-"
            f"{case.get('citation_page_end')} | {case.get('expected_page')} | "
            f"{case.get('citation_page_accuracy')} | {case.get('error')} |"
        )
    lines.extend(
        [
            "",
            "Synthetic PDFs are non-sensitive local fixtures retained under `artifacts/` "
            "for audit. This script exercises the Docker API upload, parse, index, "
            "retrieval, and citation-page mapping path; it does not run the formal "
            "50-item QA evaluation.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
