import argparse
import hashlib
import json
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the public arXiv audit corpus.")
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/evaluation/paper_audit_manifest.json")
    )
    parser.add_argument("--output", type=Path, default=Path("data/raw/audit"))
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    registry: list[dict[str, str | int]] = []
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        for entry in entries:
            arxiv_id = entry["arxiv_id"]
            url = f"https://arxiv.org/pdf/{arxiv_id}"
            destination = args.output / f"{arxiv_id}.pdf"
            response = client.get(url)
            response.raise_for_status()
            payload = response.content
            if not payload.startswith(b"%PDF-"):
                raise RuntimeError(f"arXiv did not return a PDF for {arxiv_id}")
            destination.write_bytes(payload)
            registry.append(
                {
                    **entry,
                    "source_url": url,
                    "path": str(destination),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            print(f"downloaded {arxiv_id}: {len(payload)} bytes")
    (args.output / "source_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
