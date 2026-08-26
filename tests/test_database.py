import tempfile
import unittest
from pathlib import Path

from uniguide.database import RagDatabase
from uniguide.models import TextChunk


class DatabaseTests(unittest.TestCase):
    def test_replace_document_and_stats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = RagDatabase(root / "test.db")
            chunk = TextChunk(
                source_path=root / "rules.md",
                source_name="rules.md",
                chunk_index=0,
                page_number=None,
                content="Yandal için gerekli ortalama 65'tir.",
            )
            database.replace_document(
                "rules.md", "rules.md", "hash-1", [chunk], [[1.0, 0.0]]
            )
            self.assertEqual(database.stats(), (1, 1))
            self.assertTrue(database.is_current("rules.md", "hash-1"))
            stored = database.all_chunks()[0]
            self.assertEqual(stored.source_name, "rules.md")
            self.assertEqual(stored.embedding, [1.0, 0.0])

    def test_replacing_document_does_not_duplicate_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = RagDatabase(root / "test.db")
            chunk = TextChunk(root / "a.md", "a.md", 0, None, "İlk metin")
            database.replace_document("a.md", "a.md", "hash-1", [chunk], [[1.0]])
            changed = TextChunk(root / "a.md", "a.md", 0, None, "Güncel metin")
            database.replace_document(
                "a.md", "a.md", "hash-2", [changed], [[2.0]]
            )
            self.assertEqual(database.stats(), (1, 1))
            self.assertEqual(database.all_chunks()[0].content, "Güncel metin")


if __name__ == "__main__":
    unittest.main()
