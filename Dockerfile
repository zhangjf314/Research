FROM python:3.12-slim AS runtime

ARG APP_VERSION=0.9.0rc3
LABEL org.opencontainers.image.version=$APP_VERSION
LABEL org.opencontainers.image.title="paper-research-agent"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY migrations ./migrations
COPY alembic.ini ./

RUN pip install --upgrade pip && pip install .

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/raw /app/data/parsed /app/data/reports \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/').read()"

CMD ["uvicorn", "paper_research.main:app", "--host", "0.0.0.0", "--port", "8000"]
