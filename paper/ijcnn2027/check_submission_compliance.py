#!/usr/bin/env python3
"""Check the repository-verifiable parts of the IJCNN submission contract."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from pypdf import PdfReader


LETTER = (612.0, 792.0)
A4 = (595.276, 841.89)
SIZE_TOLERANCE_PT = 1.0


def fail(message: str) -> None:
    raise AssertionError(message)


def run(*args: str) -> str:
    completed = subprocess.run(
        args,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout


def close_size(actual: tuple[float, float], expected: tuple[float, float]) -> bool:
    return all(abs(a - e) <= SIZE_TOLERANCE_PT for a, e in zip(actual, expected))


def classify_page_size(size: tuple[float, float]) -> str:
    if close_size(size, LETTER):
        return "US Letter"
    if close_size(size, A4):
        return "A4"
    fail(f"unexpected page size {size[0]:.3f} x {size[1]:.3f} pt")


def check_tex(tex_path: Path) -> None:
    source = tex_path.read_text(encoding="utf-8")
    required = {
        r"\documentclass[conference]{IEEEtran}": "IEEE conference class",
        r"\author{\IEEEauthorblockN{Anonymous Authors}}": "anonymous author block",
        r"\bibliographystyle{IEEEtran}": "IEEE bibliography style",
        r"\section*{Acknowledgment}": "AI-use acknowledgment",
        "OpenAI ChatGPT and Codex": "AI systems",
        "authors verified": "human verification statement",
    }
    for needle, label in required.items():
        if needle not in source:
            fail(f"missing {label}: {needle}")

    forbidden_patterns = {
        r"\\IEEEauthorblockA\s*\{": "affiliation block",
        r"\\thanks\s*\{": "author footnote",
        r"\\hypersetup\b": "PDF metadata override",
        r"\\usepackage(?:\[[^]]*\])?\{hyperref\}": "hyperref package",
        r"\\usepackage(?:\[[^]]*\])?\{geometry\}": "geometry override",
        r"github\.com/j-cap/2026_NeuromorphicFL": "identity-revealing repository URL",
        r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}": "email address",
    }
    for pattern, label in forbidden_patterns.items():
        if re.search(pattern, source, flags=re.IGNORECASE):
            fail(f"anonymous source contains {label}")


def check_fonts(pdf_path: Path) -> int:
    output = run("pdffonts", str(pdf_path))
    lines = [line for line in output.splitlines()[2:] if line.strip()]
    if not lines:
        fail("pdffonts returned no fonts")
    for line in lines:
        fields = line.split()
        if len(fields) < 8:
            fail(f"could not parse pdffonts row: {line}")
        # The final five fields are emb, sub, uni, object ID, and generation.
        embedded, subset = fields[-5], fields[-4]
        font_type = " ".join(fields[1:-7])
        if embedded != "yes" or subset != "yes":
            fail(f"font is not embedded and subset: {line}")
        if "Type 3" in font_type:
            fail(f"Type 3 font found: {line}")
    return len(lines)


def check_pdf(pdf_path: Path, max_pages: int) -> tuple[int, str, int, str]:
    reader = PdfReader(pdf_path)
    if reader.is_encrypted:
        fail("PDF is encrypted")
    if not 1 <= len(reader.pages) <= max_pages:
        fail(f"PDF has {len(reader.pages)} pages; maximum is {max_pages}")

    metadata = reader.metadata or {}
    if str(metadata.get("/Author", "")).strip():
        fail("PDF metadata contains an author identity")
    root = reader.root_object
    if root.get("/Outlines") is not None:
        fail("PDF contains bookmarks/outlines")
    names = root.get("/Names")
    if names is not None and names.get_object().get("/EmbeddedFiles") is not None:
        fail("PDF contains embedded files")

    sizes: list[tuple[float, float]] = []
    for index, page in enumerate(reader.pages, start=1):
        if int(page.get("/Rotate", 0)) != 0:
            fail(f"page {index} is rotated")
        if page.get("/Annots"):
            fail(f"page {index} contains annotations or links")
        box = page.mediabox
        size = (float(box.width), float(box.height))
        sizes.append(size)
        if page.cropbox != page.mediabox:
            fail(f"page {index} crop box differs from media box")
    if any(not close_size(size, sizes[0]) for size in sizes[1:]):
        fail("PDF pages do not have a uniform size")
    page_size = classify_page_size(sizes[0])

    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    for needle in ("Anonymous Authors", "ACKNOWLEDGMENT", "OpenAI ChatGPT and Codex"):
        if needle.casefold() not in text.casefold():
            fail(f"rendered PDF is missing {needle!r}")
    if re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, re.IGNORECASE):
        fail("rendered PDF contains an email address")
    if "github.com/j-cap/2026_NeuromorphicFL" in text:
        fail("rendered PDF contains the public project URL")

    info = run("pdfinfo", str(pdf_path))
    version_match = re.search(r"^PDF version:\s*([0-9.]+)$", info, re.MULTILINE)
    if not version_match:
        fail("pdfinfo did not report a PDF version")
    version = version_match.group(1)
    if tuple(map(int, version.split("."))) < (1, 4):
        fail(f"PDF version {version} is older than 1.4")

    font_count = check_fonts(pdf_path)
    return len(reader.pages), page_size, font_count, version


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", type=Path, default=Path("main.tex"))
    parser.add_argument("--pdf", type=Path, default=Path("main.pdf"))
    parser.add_argument("--max-pages", type=int, default=6)
    args = parser.parse_args()

    try:
        check_tex(args.tex)
        pages, page_size, fonts, version = check_pdf(args.pdf, args.max_pages)
    except (AssertionError, OSError, subprocess.CalledProcessError) as error:
        print(f"SUBMISSION COMPLIANCE FAILED: {error}", file=sys.stderr)
        return 1

    print(
        "Submission compliance checks passed: "
        f"anonymous source/PDF, {pages} pages, {page_size}, PDF {version}, "
        f"{fonts} embedded subset fonts, no links/bookmarks/attachments."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
