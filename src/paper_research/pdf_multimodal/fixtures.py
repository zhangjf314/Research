"""Synthetic PDFs only; no canonical-corpus examples or rule tuning."""

from pathlib import Path

import pymupdf as fitz

from .identity import file_sha


def make_fixtures(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(1, 13):
        doc = fitz.open()
        p = doc.new_page(width=400, height=500)
        if i == 1:
            p.insert_text((40, 60), "Exact native text: Alpha 123.", fontsize=12)
        if i in (2, 8, 9):
            pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40), False)
            pix.clear_with(170)
            box = fitz.Rect(50, 100, 300, 260) if i != 8 else fitz.Rect(10, 10, 390, 490)
            xref = p.insert_image(box, stream=pix.tobytes("png"), keep_proportion=False)
            if i == 9:
                p.insert_image(fitz.Rect(50, 300, 300, 450), xref=xref, keep_proportion=False)
            if i == 2:
                p.insert_text((50, 285), "Figure 1: Raster fixture", fontsize=12)
        if i == 3:
            p.draw_rect(fitz.Rect(50, 100, 300, 250), color=(0, 0, 1), fill=(0.9, 0.9, 1), width=2)
            p.insert_text((50, 275), "Fig. 2: Vector fixture", fontsize=12)
        if i in (4, 10, 12):
            if i in (4, 12):
                p.insert_text((50, 75), "Table 1: Grid fixture", fontsize=12)
            for x in (50, 150, 250):
                p.draw_line((x, 100), (x, 200))
            for y in (100, 150, 200):
                p.draw_line((50, y), (250, y))
            for x, y, t in [(60, 125, "A"), (160, 125, "B"), (60, 175, "1"), (160, 175, "2")]:
                p.insert_text((x, y), t, fontsize=12)
        if i == 5:
            p.insert_text((50, 150), "x + y = z (1)", fontsize=16)
        if i == 6:
            p.insert_text((50, 150), "x+y=z", fontsize=16)
        if i == 7:
            p.insert_text((40, 150), "The label = a name used in ordinary prose.", fontsize=12)
        if i == 11:
            p.insert_text((60, 120), "Rotated crop fixture", fontsize=12)
            p.set_cropbox(fitz.Rect(20, 30, 380, 470))
            p.set_rotation(90)
        if i == 12:
            p.set_cropbox(fitz.Rect(20, 30, 380, 470))
            p.set_rotation(90)
        path = directory / f"F{i:02d}.pdf"
        doc.set_metadata({})
        doc.save(path, garbage=0, deflate=False, no_new_id=True)
        doc.close()
        records.append(
            {
                "fixture": f"F{i:02d}",
                "paper_id": f"synthetic:F{i:02d}",
                "path": path.name,
                "pdf_sha256": file_sha(path),
            }
        )
    return records
