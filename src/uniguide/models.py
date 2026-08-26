from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocumentPage:
    source_path: Path
    source_name: str
    page_number: int | None
    text: str


@dataclass(frozen=True)
class TextChunk:
    source_path: Path
    source_name: str
    chunk_index: int
    page_number: int | None
    content: str


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: int
    source_name: str
    source_path: str
    chunk_index: int
    page_number: int | None
    content: str
    embedding: list[float]


@dataclass(frozen=True)
class SearchResult:
    chunk_id: int
    source_name: str
    source_path: str
    chunk_index: int
    page_number: int | None
    content: str
    score: float

    @property
    def citation(self) -> str:
        page = f", sayfa {self.page_number}" if self.page_number else ""
        return f"{self.source_name}{page}"


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    sources: list[SearchResult]
    grounded: bool


@dataclass(frozen=True)
class IndexReport:
    discovered_documents: int
    indexed_documents: int
    skipped_documents: int
    indexed_chunks: int
