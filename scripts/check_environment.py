import importlib.util
import json
import shutil
import sys


def main() -> None:
    checks = {
        "python_3_12": sys.version_info[:2] == (3, 12),
        "docker_cli": shutil.which("docker") is not None,
        "tesseract_optional": shutil.which("tesseract") is not None,
        "pymupdf": importlib.util.find_spec("fitz") is not None,
        "langgraph": importlib.util.find_spec("langgraph") is not None,
    }
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    required = ["python_3_12", "docker_cli", "pymupdf", "langgraph"]
    if not all(checks[name] for name in required):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
