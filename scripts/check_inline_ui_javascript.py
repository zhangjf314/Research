from __future__ import annotations

# ruff: noqa: E402,I001
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_ui_javascript_syntax_v1 import main


if __name__ == "__main__":
    sys.exit(main())
