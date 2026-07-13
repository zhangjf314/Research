import json
import threading
from datetime import UTC, datetime
from pathlib import Path


class IndexRegistry:
    _lock = threading.Lock()

    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, logical_collection: str) -> str:
        data = self.read()
        return str(data.get("defaults", {}).get(logical_collection, logical_collection))

    def switch(
        self,
        logical_collection: str,
        physical_collection: str,
        metadata: dict[str, object],
    ) -> None:
        with self._lock:
            data = self.read()
            data.setdefault("defaults", {})[logical_collection] = physical_collection
            data.setdefault("indexes", {})[physical_collection] = {
                **metadata,
                "activated_at": datetime.now(UTC).isoformat(),
            }
            self._write(data)

    def read(self) -> dict:
        if not self.path.exists():
            return {"defaults": {}, "indexes": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)
