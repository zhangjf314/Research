from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from paper_research.config import Settings


@contextmanager
def checkpoint_saver(settings: Settings) -> Iterator[BaseCheckpointSaver]:
    """Return a profile-appropriate saver without silently changing providers."""
    if settings.checkpoint_provider == "memory":
        yield InMemorySaver()
        return
    if settings.checkpoint_provider != "postgres":
        raise ValueError(f"unsupported checkpoint provider: {settings.checkpoint_provider}")
    if not settings.checkpoint_database_url:
        raise ValueError("CHECKPOINT_DATABASE_URL is required for postgres checkpoints")

    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(settings.checkpoint_database_url) as saver:
        saver.setup()
        yield saver
