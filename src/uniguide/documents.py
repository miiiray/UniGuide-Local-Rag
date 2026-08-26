from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

from uniguide.models import DocumentPage, TextChunk


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def discover_documents(data_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and not path.name.startswith(".")
        and path.name.upper() not in {"README.MD", "SOURCES.MD"}
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_document(path: Path) -> list[DocumentPage]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages: list[DocumentPage] = []
        for index, page in enumerate(reader.pages, start=1):
            text = normalize_text(page.extract_text() or "")
            if text:
                pages.append(
                    DocumentPage(
                        source_path=path,
                        source_name=path.name,
                        page_number=index,
                        text=text,
                    )
                )
        return pages

    if suffix in {".txt", ".md"}:
        text = normalize_text(path.read_text(encoding="utf-8", errors="replace"))
        return [
            DocumentPage(
                source_path=path,
                source_name=path.name,
                page_number=None,
                text=text,
            )
        ] if text else []

    raise ValueError(f"Desteklenmeyen dosya türü: {path.suffix}")


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 180) -> list[str]:
    if max_chars < 200:
        raise ValueError("max_chars en az 200 olmalıdır.")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap, 0 ile max_chars arasında olmalıdır.")

    text = normalize_text(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        proposed_end = min(start + max_chars, len(text))
        end = proposed_end

        if proposed_end < len(text):
            search_floor = start + int(max_chars * 0.6)
            candidates = [
                text.rfind("\n\n", search_floor, proposed_end),
                text.rfind(". ", search_floor, proposed_end),
                text.rfind("\n", search_floor, proposed_end),
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + (2 if text[boundary : boundary + 2] in {"\n\n", ". "} else 1)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break

        next_start = max(0, end - overlap)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def build_chunks(
    pages: Iterable[DocumentPage], max_chars: int = 1200, overlap: int = 180
) -> list[TextChunk]:
    result: list[TextChunk] = []
    chunk_index = 0
    for page in pages:
        for content in chunk_text(page.text, max_chars=max_chars, overlap=overlap):
            result.append(
                TextChunk(
                    source_path=page.source_path,
                    source_name=page.source_name,
                    chunk_index=chunk_index,
                    page_number=page.page_number,
                    content=content,
                )
            )
            chunk_index += 1
    return result
