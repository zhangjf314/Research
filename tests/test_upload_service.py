import hashlib
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import UploadFile

from paper_research.config import Settings
from paper_research.ingestion.upload_service import UploadService, UploadValidationError


def service(tmp_path: Path, max_bytes: int = 1024) -> UploadService:
    settings = Settings(data_dir=tmp_path, max_upload_bytes=max_bytes)
    return UploadService(session=None, settings=settings)  # type: ignore[arg-type]


def test_copy_and_hash_streams_upload(tmp_path: Path) -> None:
    payload = b"%PDF-fixture"
    upload = UploadFile(filename="paper.pdf", file=BytesIO(payload))
    destination = tmp_path / "upload.part"

    digest, size = service(tmp_path)._copy_and_hash(upload, destination)

    assert digest == hashlib.sha256(payload).hexdigest()
    assert size == len(payload)
    assert destination.read_bytes() == payload


def test_upload_size_limit_is_enforced(tmp_path: Path) -> None:
    upload = UploadFile(filename="paper.pdf", file=BytesIO(b"12345"))

    with pytest.raises(UploadValidationError, match="size limit"):
        service(tmp_path, max_bytes=4)._copy_and_hash(upload, tmp_path / "upload.part")


@pytest.mark.parametrize("filename", [None, "paper.txt", "paper"])
def test_only_pdf_filename_is_accepted(filename: str | None) -> None:
    with pytest.raises(UploadValidationError, match="only .pdf"):
        UploadService._validate_filename(filename)
