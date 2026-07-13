"""create papers table"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

paper_status = sa.Enum(
    "UPLOADED", "PARSING", "PARSED", "INDEXING", "READY", "FAILED", name="paper_status"
)


def upgrade() -> None:
    op.create_table(
        "papers",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.String(1000), nullable=False),
        sa.Column("authors", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column("abstract", sa.Text()),
        sa.Column("keywords", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column("year", sa.Integer()),
        sa.Column("venue", sa.String(500)),
        sa.Column("doi", sa.String(255), unique=True),
        sa.Column("arxiv_id", sa.String(64), unique=True),
        sa.Column("source_type", sa.String(32), nullable=False),
        sa.Column("source_url", sa.Text()),
        sa.Column("pdf_path", sa.Text()),
        sa.Column("file_hash", sa.String(64), unique=True),
        sa.Column("language", sa.String(16)),
        sa.Column("parse_status", paper_status, nullable=False),
        sa.Column("index_status", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_papers_title", "papers", ["title"])
    op.create_index("ix_papers_year", "papers", ["year"])
    op.create_index("ix_papers_file_hash", "papers", ["file_hash"])


def downgrade() -> None:
    op.drop_table("papers")
    paper_status.drop(op.get_bind(), checkfirst=True)
