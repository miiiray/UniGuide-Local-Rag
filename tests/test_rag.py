import tempfile
import unittest
from pathlib import Path

from uniguide.config import Settings
from uniguide.models import TextChunk
from uniguide.rag import RagService, cosine_similarity


class FakeRuntime:
    def embed_one(self, text: str) -> list[float]:
        return [1.0, 0.0] if "yandal" in text.lower() else [0.0, 1.0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(text) for text in texts]

    def complete(self, messages: list[dict[str, str]]) -> str:
        if "KAYNAK METİN" not in messages[1]["content"]:
            raise AssertionError("Prompt içinde KAYNAK METİN bulunamadı")
        return "Yandal için gereken ortalama 65'tir."


class HallucinatingRuntime(FakeRuntime):
    def complete(self, messages: list[dict[str, str]]) -> str:
        return "En az dört dil öğrenme sürecinden birkaç ay sonra kabul edilir."


class WrongOrdinalRuntime(FakeRuntime):
    def complete(self, messages: list[dict[str, str]]) -> str:
        return (
            "Yandal programına en erken dördüncü, en geç altıncı yarıyılın "
            "başında başvurulabilir."
        )


def settings_for(root: Path, threshold: float = 0.35) -> Settings:
    return Settings(
        data_dir=root / "data",
        db_path=root / "rag.db",
        embedding_model="fake-embedding",
        chat_model="fake-chat",
        top_k=2,
        min_similarity=threshold,
        chunk_size=400,
        chunk_overlap=40,
    )


def seed(service: RagService, root: Path) -> None:
    chunks = [
        TextChunk(root / "yandal.md", "yandal.md", 0, None, "Yandal ortalaması 65."),
        TextChunk(root / "staj.md", "staj.md", 0, None, "Staj süresi 60 iş günüdür."),
    ]
    service.database.replace_document(
        "yandal.md", "yandal.md", "h1", [chunks[0]], [[1.0, 0.0]]
    )
    service.database.replace_document(
        "staj.md", "staj.md", "h2", [chunks[1]], [[0.0, 1.0]]
    )


class RagTests(unittest.TestCase):
    def test_cosine_similarity(self) -> None:
        self.assertEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_search_ranks_semantically_closest_chunk_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = RagService(settings_for(root), FakeRuntime())
            seed(service, root)
            results = service.search("Yandal koşulu nedir?")
            self.assertEqual(results[0].source_name, "yandal.md")
            self.assertEqual(results[0].score, 1.0)

    def test_ask_returns_grounded_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = RagService(settings_for(root), FakeRuntime())
            seed(service, root)
            result = service.ask("Yandal koşulu nedir?")
            self.assertTrue(result.grounded)
            self.assertIn("65", result.answer)
            self.assertIn("Kaynaklar:", result.answer)
            self.assertEqual(result.sources[0].source_name, "yandal.md")

    def test_hallucinated_answer_falls_back_to_source_sentence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = RagService(settings_for(root), HallucinatingRuntime())
            chunk = TextChunk(
                root / "yandal.md",
                "yandal.md",
                0,
                None,
                (
                    "Yandal programına anadal lisans programının en erken üçüncü, "
                    "en geç altıncı yarıyılının başında başvurulabilir."
                ),
            )
            service.database.replace_document(
                "yandal.md", "yandal.md", "h1", [chunk], [[1.0, 0.0]]
            )

            result = service.ask(
                "Yandal programına en erken ve en geç hangi yarıyılda başvurabilirim?"
            )

            self.assertIn("en erken üçüncü", result.answer)
            self.assertIn("en geç altıncı", result.answer)
            self.assertNotIn("dört dil", result.answer)
            self.assertIn("yandal.md", result.answer)

    def test_wrong_ordinal_falls_back_to_source_sentence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = RagService(settings_for(root), WrongOrdinalRuntime())
            chunk = TextChunk(
                root / "yandal.md",
                "yandal.md",
                0,
                None,
                (
                    "Yandal programına anadal lisans programının en erken üçüncü, "
                    "en geç altıncı yarıyılının başında başvurulabilir."
                ),
            )
            service.database.replace_document(
                "yandal.md", "yandal.md", "h1", [chunk], [[1.0, 0.0]]
            )

            result = service.ask(
                "Yandal programına en erken ve en geç hangi yarıyılda başvurabilirim?"
            )

            self.assertIn("en erken üçüncü", result.answer)
            self.assertNotIn("en erken dördüncü", result.answer)

    def test_low_similarity_uses_safe_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = RagService(settings_for(root, threshold=1.1), FakeRuntime())
            seed(service, root)
            result = service.ask("Yandal koşulu nedir?")
            self.assertFalse(result.grounded)
            self.assertIn("bulunamadı", result.answer)

    def test_index_documents_then_skip_unchanged_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = settings_for(root)
            settings.data_dir.mkdir(parents=True)
            (settings.data_dir / "yandal.md").write_text(
                "Yandal programına üçüncü yarıyılda başvurulabilir.", encoding="utf-8"
            )
            (settings.data_dir / "staj.txt").write_text(
                "Staj süresi 60 iş günüdür.", encoding="utf-8"
            )
            service = RagService(settings, FakeRuntime())

            first = service.index_documents()
            second = service.index_documents()

            self.assertEqual(first.indexed_documents, 2)
            self.assertEqual(first.indexed_chunks, 2)
            self.assertEqual(second.indexed_documents, 0)
            self.assertEqual(second.skipped_documents, 2)
            self.assertEqual(service.database.stats(), (2, 2))


if __name__ == "__main__":
    unittest.main()
