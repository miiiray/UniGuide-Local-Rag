import unittest
import tempfile
from pathlib import Path

from uniguide.documents import chunk_text, discover_documents, normalize_text


class DocumentTests(unittest.TestCase):
    def test_normalize_text_removes_excess_whitespace(self) -> None:
        self.assertEqual(
            normalize_text("Bir   metin\n\n\n\nİkinci bölüm"),
            "Bir metin\n\nİkinci bölüm",
        )

    def test_chunk_text_splits_long_content_with_overlap(self) -> None:
        text = " ".join(f"kelime-{index}." for index in range(500))
        chunks = chunk_text(text, max_chars=300, overlap=40)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(chunk.strip() for chunk in chunks))
        self.assertTrue(all(len(chunk) <= 302 for chunk in chunks))

    def test_chunk_text_rejects_invalid_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            chunk_text("örnek", max_chars=300, overlap=300)

    def test_official_documents_replace_demo_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            demo_dir = data_dir / "demo"
            official_dir = data_dir / "official"
            demo_dir.mkdir()
            official_dir.mkdir()
            (demo_dir / "ozet.md").write_text("Demo", encoding="utf-8")
            official_pdf = official_dir / "yonerge.pdf"
            official_pdf.write_bytes(b"%PDF-1.4")

            self.assertEqual(discover_documents(data_dir), [official_pdf])


if __name__ == "__main__":
    unittest.main()
