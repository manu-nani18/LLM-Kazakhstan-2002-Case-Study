"""Extract, clean, and segment the Kazakhstan 2002 NHDR PDF.

Outputs are written to ../outputs:
- raw_pages.json: raw text by page
- cleaned_pages.json: cleaned text by page
- cleaned_text.txt: full cleaned text with page markers
- chapter_chunks.json: detected chapter-level chunks
- processing_summary.json: quick counts for reporting/debugging
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "data" / "kazakhstan-2002-nhdr.pdf"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


@dataclass
class PageText:
    page: int
    text: str


@dataclass
class ChapterChunk:
    chapter_number: int
    title: str
    start_page: int
    end_page: int
    text: str
    word_count: int


HEADER_PATTERNS = [
    re.compile(r"^\s*Human Development Report Kazakhstan 2002\s*$", re.I),
    re.compile(r"^\s*\d+\s+Human Development Report Kazakhstan 2002\s*$", re.I),
]

CHAPTER_LINE_RE = re.compile(r"^\s*Chapter\s+(\d+)\s*\.?\s*(.+?)\s*$", re.I)

# The chapter titles are visible in the PDF/table of contents, but some heading
# text is embedded in the layout and is skipped by text extraction. These
# boundaries keep chapter chunks aligned with the actual report structure.
REPORT_CHAPTERS = [
    (1, "Rural Development in Kazakhstan", 13, 20),
    (2, "The Economy as a Factor in Sustainable Rural Human Development", 21, 26),
    (3, "Social Development in Rural Kazakhstan", 27, 48),
    (4, "A Proposed Approach to Rural Development", 49, 58),
]

MOJIBAKE_REPLACEMENTS = {
    "\u00e2\u20ac\u0153": '"',
    "\u00e2\u20ac\u009d": '"',
    "\u00e2\u20ac\u2122": "'",
    "\u00e2\u20ac\u02dc": "'",
    "\u00e2\u20ac\u201c": "-",
    "\u00e2\u20ac\u201d": "-",
    "\u00e2\u20ac\u00a6": "...",
    "\u00c2": "",
}


def normalise_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00ad", "")
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(bad, good)
    text = text.translate(str.maketrans({
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        " ": " ",
    }))
    text = text.replace("??", "n")
    text = text.replace("??", "T")
    text = text.replace("??", "A")
    text = text.replace("??", "a")
    text = text.replace("?", "K")
    text = text.replace("?", "M")
    text = text.replace("\u00f1", "c")
    text = text.replace("\u00d2", "T")
    return text


def extract_raw_pages(pdf_path: Path) -> list[PageText]:
    reader = PdfReader(str(pdf_path))
    pages: list[PageText] = []
    for index, page in enumerate(reader.pages, start=1):
        pages.append(PageText(page=index, text=page.extract_text() or ""))
    return pages


def clean_page(text: str, page_number: int) -> str:
    text = normalise_text(text)
    lines = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line == str(page_number):
            continue
        if re.fullmatch(r"\d+", line):
            continue
        if any(pattern.match(line) for pattern in HEADER_PATTERNS):
            continue
        line = re.sub(r"^\d+\s*(?=Executive Summary|Foreword|Table of contents|Chapter\s+\d+)", "", line)
        lines.append(line)

    joined = "\n".join(lines)
    joined = re.sub(r"([A-Za-z])-\n([a-z])", r"\1\2", joined)
    joined = re.sub(r"(?<![.!?:;])\n(?!\s*(Chapter\s+\d+|\d+\.\d+|Executive Summary|Foreword|Table of contents))", " ", joined)
    joined = re.sub(r"[ \t]+", " ", joined)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


def find_chapter_starts(cleaned_pages: list[PageText]) -> list[tuple[int, int, str]]:
    starts: list[tuple[int, int, str]] = []
    seen: set[int] = set()

    for page in cleaned_pages:
        if page.page < 12:
            continue
        for line in page.text.split("\n"):
            match = CHAPTER_LINE_RE.match(line)
            if not match:
                continue
            chapter_number = int(match.group(1))
            title = match.group(2).strip(" .")
            if chapter_number in seen:
                continue
            if len(title) < 4 or len(title) > 140:
                continue
            seen.add(chapter_number)
            starts.append((chapter_number, page.page, title))

    starts.sort(key=lambda item: item[1])
    return starts


def build_chapter_chunks(cleaned_pages: list[PageText]) -> list[ChapterChunk]:
    page_lookup = {page.page: page.text for page in cleaned_pages}
    chunks: list[ChapterChunk] = []

    for chapter_number, title, start_page, end_page in REPORT_CHAPTERS:
        chapter_text = "\n\n".join(
            page_lookup.get(page, "") for page in range(start_page, end_page + 1)
        ).strip()
        chunks.append(
            ChapterChunk(
                chapter_number=chapter_number,
                title=title,
                start_page=start_page,
                end_page=end_page,
                text=chapter_text,
                word_count=len(re.findall(r"\b\w+\b", chapter_text)),
            )
        )
    return chunks


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_pages = extract_raw_pages(PDF_PATH)
    cleaned_pages = [PageText(page=p.page, text=clean_page(p.text, p.page)) for p in raw_pages]
    chapters = build_chapter_chunks(cleaned_pages)

    full_text = "\n\n".join(
        f"[PAGE {page.page}]\n{page.text}" for page in cleaned_pages if page.text
    )

    write_json(OUTPUT_DIR / "raw_pages.json", [asdict(page) for page in raw_pages])
    write_json(OUTPUT_DIR / "cleaned_pages.json", [asdict(page) for page in cleaned_pages])
    write_json(OUTPUT_DIR / "chapter_chunks.json", [asdict(chapter) for chapter in chapters])
    (OUTPUT_DIR / "cleaned_text.txt").write_text(full_text, encoding="utf-8")

    summary = {
        "pdf": str(PDF_PATH),
        "page_count": len(raw_pages),
        "cleaned_word_count": len(re.findall(r"\b\w+\b", full_text)),
        "chapter_count": len(chapters),
        "chapters": [
            {
                "chapter_number": chapter.chapter_number,
                "title": chapter.title,
                "start_page": chapter.start_page,
                "end_page": chapter.end_page,
                "word_count": chapter.word_count,
            }
            for chapter in chapters
        ],
    }
    write_json(OUTPUT_DIR / "processing_summary.json", summary)

    print(f"Extracted {summary['page_count']} pages")
    print(f"Cleaned word count: {summary['cleaned_word_count']}")
    print(f"Detected {summary['chapter_count']} chapters")
    for chapter in summary["chapters"]:
        print(
            f"Chapter {chapter['chapter_number']}: {chapter['title']} "
            f"(pages {chapter['start_page']}-{chapter['end_page']}, {chapter['word_count']} words)"
        )


if __name__ == "__main__":
    main()





