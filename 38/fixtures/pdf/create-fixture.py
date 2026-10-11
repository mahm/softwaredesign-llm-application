"""Create a wholly synthetic, fixed-answer two-page PDF and image-only copy.

Run with: uv run --with reportlab --with pypdf create-fixture.py
"""
from pathlib import Path
import hashlib
import json
import subprocess
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from pypdf import PdfReader

root = Path(__file__).resolve().parent
width, height = A4
expected = {
    "quoteId": "SD38-042", "totalYen": 2240,
    "validUntil": "2026-10-31", "destination": "Meeting Room B",
}
text_path = root / "quote-text.pdf"
c = canvas.Canvas(str(text_path), pagesize=A4, invariant=1)
c.setTitle("Synthetic quote SD38-042")
for page in (1, 2):
    c.setFillColor(colors.HexColor("#15304b"))
    c.setFont("Helvetica-Bold", 23)
    c.drawString(52, height - 68, "SYNTHETIC QUOTE")
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor("#566579"))
    c.drawString(52, height - 91, "OpenRouter PDF fixture - fictional data for testing")
    c.setStrokeColor(colors.HexColor("#bdd0e1"))
    c.line(52, height - 110, width - 52, height - 110)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(52, height - 145, "Quote ID: SD38-042")
    c.setFont("Helvetica", 13)
    if page == 1:
        rows = [
            "Notebook: 3 units x 480 JPY = 1,440 JPY",
            "Pen: 5 units x 160 JPY = 800 JPY",
        ]
        for index, row in enumerate(rows):
            c.drawString(52, height - 199 - index * 34, row)
        c.setFont("Helvetica-Bold", 17)
        c.drawString(52, height - 300, "TOTAL: 2,240 JPY")
        c.setFont("Helvetica", 11)
        c.drawString(52, height - 336, "Prices are final; no additional tax or shipping charge.")
    else:
        c.drawString(52, height - 199, "Valid until: 2026-10-31")
        c.drawString(52, height - 237, "Delivery destination: Meeting Room B")
        c.drawString(52, height - 290, "Reference code: COBALT-73")
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor("#566579"))
    c.drawString(52, 42, f"SD38-042 | Page {page} of 2 | Synthetic fixture")
    c.showPage()
c.save()
subprocess.run(["pdftoppm", "-r", "120", "-png", str(text_path), str(root / "quote-text")], check=True)
scan_path = root / "quote-scan.pdf"
scan = canvas.Canvas(str(scan_path), pagesize=A4, invariant=1)
scan.setTitle("Synthetic quote SD38-042 - image-only")
for page in (1, 2):
    scan.drawImage(str(root / f"quote-text-{page}.png"), 0, 0, width=width, height=height)
    scan.showPage()
scan.save()
subprocess.run(["pdftoppm", "-r", "120", "-png", str(scan_path), str(root / "quote-scan")], check=True)
text_reader = PdfReader(text_path)
scan_reader = PdfReader(scan_path)
extracted = "\n".join(page.extract_text() or "" for page in text_reader.pages)
assert len(text_reader.pages) == len(scan_reader.pages) == 2
for needle in ("SD38-042", "2,240", "2026-10-31", "Meeting Room B", "COBALT-73"):
    assert needle in extracted, needle
assert all(not page.extract_text() for page in scan_reader.pages), "Scan copy contains selectable text"
manifest = {
    "expected": expected, "followupExpected": {"destination": "Meeting Room B", "referenceCode": "COBALT-73"},
    "pages": 2, "textChecksPassed": True, "scanHasNoTextLayer": True,
    "files": {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
              for p in (text_path, scan_path)},
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
