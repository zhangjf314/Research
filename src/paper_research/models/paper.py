import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column

from paper_research.db import Base


class PaperStatus(enum.StrEnum):
    uploaded = "UPLOADED"
    parsing = "PARSING"
    parsed = "PARSED"
    indexing = "INDEXING"
    ready = "READY"
    failed = "FAILED"


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(1000), nullable=False, index=True)
    authors: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    abstract: Mapped[str | None] = mapped_column(Text)
    keywords: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    year: Mapped[int | None] = mapped_column(Integer, index=True)
    venue: Mapped[str | None] = mapped_column(String(500))
    doi: Mapped[str | None] = mapped_column(String(255), unique=True)
    arxiv_id: Mapped[str | None] = mapped_column(String(64), unique=True)
    source_type: Mapped[str] = mapped_column(String(32), default="upload")
    source_url: Mapped[str | None] = mapped_column(Text)
    pdf_path: Mapped[str | None] = mapped_column(Text)
    file_hash: Mapped[str | None] = mapped_column(String(64), unique=True, index=True)
    language: Mapped[str | None] = mapped_column(String(16))
    parse_status: Mapped[PaperStatus] = mapped_column(
        Enum(
            PaperStatus,
            name="paper_status",
            values_callable=lambda enum_type: [item.value for item in enum_type],
        ),
        default=PaperStatus.uploaded,
    )
    index_status: Mapped[str] = mapped_column(String(32), default="PENDING")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
