import unittest

from uniguide.documents import chunk_text, normalize_text


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


if __name__ == "__main__":
    unittest.main()
