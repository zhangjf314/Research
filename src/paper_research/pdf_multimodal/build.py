"""Protocol-only CLI: formal corpus stays fail-closed."""

import argparse
import socket

from .contract import make_contract
from .extraction import extract_fixture
from .fixtures import make_fixtures
from .identity import canonical


def deny_network(*args, **kwargs):
    raise RuntimeError("ZERO_NETWORK_PROTOCOL")


def network_guard():
    socket.create_connection = socket.getaddrinfo = deny_network
    socket.socket.connect = socket.socket.connect_ex = socket.socket.sendto = deny_network


def main():
    network_guard()
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-fixtures")
    parser.add_argument("--validate-protocol", action="store_true")
    parser.add_argument("--full-corpus", action="store_true")
    args = parser.parse_args()
    if args.full_corpus:
        parser.exit(2, "FORMAL_EXTRACTION_NOT_AUTHORIZED\n")
    contract = make_contract()
    if args.synthetic_fixtures:
        from pathlib import Path

        directory = Path(args.synthetic_fixtures).resolve()
        root = Path(__file__).resolve().parents[3]
        if (
            not directory.is_relative_to(root / "artifacts/paperresearch-v2/phase4a-r2-protocol")
            or (
                root / "artifacts/paperresearch-v2/phase4a-r2-protocol/completion_freeze.json"
            ).exists()
        ):
            parser.exit(2, "FIXTURE_WRITE_OUTSIDE_OPEN_PROTOCOL\n")
        records = make_fixtures(directory)
        allowed = {r["pdf_sha256"] for r in records}
        for r in records:
            extract_fixture(
                directory / r["path"], r["paper_id"], r["pdf_sha256"], contract, allowed
            )
    if args.validate_protocol:
        print(canonical(contract))


if __name__ == "__main__":
    main()
