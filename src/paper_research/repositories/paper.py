import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from paper_research.models.paper import Paper
from paper_research.schemas.paper import PaperCreate


class PaperRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, payload: PaperCreate) -> Paper:
        paper = Paper(**payload.model_dump())
        self.session.add(paper)
        self.session.commit()
        self.session.refresh(paper)
        return paper

    def list(self, *, limit: int = 50, offset: int = 0) -> list[Paper]:
        statement = select(Paper).order_by(Paper.created_at.desc()).limit(limit).offset(offset)
        return list(self.session.scalars(statement))

    def get(self, paper_id: uuid.UUID) -> Paper | None:
        return self.session.get(Paper, paper_id)

    def get_by_hash(self, file_hash: str) -> Paper | None:
        return self.session.scalar(select(Paper).where(Paper.file_hash == file_hash))

    def save(self, paper: Paper) -> Paper:
        self.session.add(paper)
        self.session.commit()
        self.session.refresh(paper)
        return paper
