from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

from uniguide.models import StoredChunk, TextChunk


class RagDatabase:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT NOT NULL UNIQUE,
                    source_name TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    indexed_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    page_number INTEGER,
                    content TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
                    UNIQUE(document_id, chunk_index)
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                ON chunks(document_id);
                """
            )

    def is_current(self, source_path: str, sha256: str) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT sha256 FROM documents WHERE source_path = ?", (source_path,)
            ).fetchone()
        return bool(row and row["sha256"] == sha256)

    def replace_document(
        self,
        source_path: str,
        source_name: str,
        sha256: str,
        chunks: Sequence[TextChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk ve embedding sayıları eşit olmalıdır.")

        with self.connect() as connection:
            existing = connection.execute(
                "SELECT id FROM documents WHERE source_path = ?", (source_path,)
            ).fetchone()
            if existing:
                document_id = int(existing["id"])
                connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                connection.execute(
                    """
                    UPDATE documents
                    SET source_name = ?, sha256 = ?, indexed_at = ?
                    WHERE id = ?
                    """,
                    (
                        source_name,
                        sha256,
                        datetime.now(timezone.utc).isoformat(),
                        document_id,
                    ),
                )
            else:
                cursor = connection.execute(
                    """
                    INSERT INTO documents(source_path, source_name, sha256, indexed_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        source_path,
                        source_name,
                        sha256,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                document_id = int(cursor.lastrowid)

            connection.executemany(
                """
                INSERT INTO chunks(
                    document_id, chunk_index, page_number, content, embedding_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        document_id,
                        chunk.chunk_index,
                        chunk.page_number,
                        chunk.content,
                        json.dumps(list(embedding), separators=(",", ":")),
                    )
                    for chunk, embedding in zip(chunks, embeddings, strict=True)
                ],
            )

    def all_chunks(self) -> list[StoredChunk]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT c.id, d.source_name, d.source_path, c.chunk_index,
                       c.page_number, c.content, c.embedding_json
                FROM chunks AS c
                JOIN documents AS d ON d.id = c.document_id
                ORDER BY d.source_name, c.chunk_index
                """
            ).fetchall()
        return [
            StoredChunk(
                chunk_id=int(row["id"]),
                source_name=str(row["source_name"]),
                source_path=str(row["source_path"]),
                chunk_index=int(row["chunk_index"]),
                page_number=int(row["page_number"]) if row["page_number"] else None,
                content=str(row["content"]),
                embedding=[float(value) for value in json.loads(row["embedding_json"])],
            )
            for row in rows
        ]

    def stats(self) -> tuple[int, int]:
        with self.connect() as connection:
            documents = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
            chunks = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        return documents, chunks

    def clear(self) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM documents")
