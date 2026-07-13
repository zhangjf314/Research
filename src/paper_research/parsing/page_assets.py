from pathlib import Path

import fitz


def render_page_assets(pdf_path: Path, output_dir: Path, dpi: int = 144) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    scale = dpi / 72
    matrix = fitz.Matrix(scale, scale)
    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document):
            destination = output_dir / f"page-{index + 1:04d}.png"
            page.get_pixmap(matrix=matrix, alpha=False).save(destination)
            paths.append(destination)
    return paths
