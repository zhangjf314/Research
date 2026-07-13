import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime
from xml.etree import ElementTree

from paper_research.search.http import CachedRetryClient
from paper_research.search.models import PaperCandidate, SearchRequest


class PaperSourceClient(ABC):
    name: str

    @abstractmethod
    def search(self, query: str, request: SearchRequest) -> list[PaperCandidate]:
        """Search a paper source."""


class ArxivClient(PaperSourceClient):
    name = "arxiv"
    endpoint = "https://export.arxiv.org/api/query"

    def __init__(self, http: CachedRetryClient) -> None:
        self.http = http
        self._lock = threading.Lock()
        self._last_request_at = 0.0

    def search(self, query: str, request: SearchRequest) -> list[PaperCandidate]:
        with self._lock:
            wait_seconds = 3.0 - (time.monotonic() - self._last_request_at)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            text = self.http.get_text(
                self.endpoint,
                params={
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": request.limit,
                    "sortBy": "relevance",
                },
            )
            self._last_request_at = time.monotonic()
        root = ElementTree.fromstring(text)
        namespace = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        candidates = []
        for entry in root.findall("atom:entry", namespace):
            identifier = self._text(entry.find("atom:id", namespace)).rsplit("/", 1)[-1]
            arxiv_id = identifier.split("v", 1)[0]
            published_text = self._text(entry.find("atom:published", namespace))
            published = datetime.fromisoformat(published_text.replace("Z", "+00:00")).date()
            doi = self._text(entry.find("arxiv:doi", namespace)) or None
            candidates.append(
                PaperCandidate(
                    source=self.name,
                    source_id=identifier,
                    title=self._clean(self._text(entry.find("atom:title", namespace))),
                    abstract=self._clean(self._text(entry.find("atom:summary", namespace))),
                    authors=[
                        self._text(author.find("atom:name", namespace))
                        for author in entry.findall("atom:author", namespace)
                    ],
                    year=published.year,
                    publication_date=published,
                    venue=self._text(entry.find("arxiv:journal_ref", namespace)) or None,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    source_url=f"https://arxiv.org/abs/{arxiv_id}",
                    pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                    is_open_access=True,
                )
            )
        return candidates

    @staticmethod
    def _text(element: ElementTree.Element | None) -> str:
        return "".join(element.itertext()).strip() if element is not None else ""

    @staticmethod
    def _clean(value: str) -> str:
        return " ".join(value.split())


class SemanticScholarClient(PaperSourceClient):
    name = "semantic_scholar"
    endpoint = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, http: CachedRetryClient, api_key: str | None = None) -> None:
        self.http = http
        self.api_key = api_key

    def search(self, query: str, request: SearchRequest) -> list[PaperCandidate]:
        fields = (
            "title,abstract,authors,year,venue,externalIds,url,citationCount,"
            "openAccessPdf,publicationDate"
        )
        headers = {"x-api-key": self.api_key} if self.api_key else None
        payload = self.http.get_json(
            self.endpoint,
            params={"query": query.replace("-", " "), "limit": request.limit, "fields": fields},
            headers=headers,
        )
        candidates = []
        for item in payload.get("data", []):
            external = item.get("externalIds") or {}
            open_pdf = item.get("openAccessPdf") or {}
            publication_date = item.get("publicationDate")
            candidates.append(
                PaperCandidate(
                    source=self.name,
                    source_id=item["paperId"],
                    title=item.get("title") or "Untitled",
                    abstract=item.get("abstract"),
                    authors=[author.get("name", "") for author in item.get("authors") or []],
                    year=item.get("year"),
                    publication_date=publication_date,
                    venue=item.get("venue"),
                    doi=external.get("DOI"),
                    arxiv_id=external.get("ArXiv"),
                    source_url=item.get("url"),
                    pdf_url=open_pdf.get("url"),
                    citation_count=item.get("citationCount") or 0,
                    is_open_access=bool(open_pdf.get("url")),
                )
            )
        return candidates
