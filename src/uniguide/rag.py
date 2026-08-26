from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable, Sequence

from uniguide.config import Settings
from uniguide.database import RagDatabase
from uniguide.documents import (
    build_chunks,
    discover_documents,
    file_sha256,
    read_document,
)
from uniguide.models import IndexReport, RagAnswer, SearchResult


_WORD_PATTERN = re.compile(r"[a-zA-Z0-9çğıöşüÇĞİÖŞÜ]+")
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
_NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")
_NUMBER_WORDS = {
    "iki",
    "uc",
    "dort",
    "bes",
    "alti",
    "yedi",
    "sekiz",
    "dokuz",
    "on",
    "birinci",
    "ikinci",
    "ucuncu",
    "dorduncu",
    "besinci",
    "altinci",
    "yedinci",
    "sekizinci",
    "dokuzuncu",
    "onuncu",
}
_STOP_WORDS = {
    "acaba",
    "ama",
    "ancak",
    "bir",
    "bu",
    "da",
    "de",
    "daha",
    "en",
    "gibi",
    "hangi",
    "icin",
    "ile",
    "mi",
    "mu",
    "mı",
    "mü",
    "ne",
    "neden",
    "nasil",
    "ve",
    "veya",
}


def _normalize_word(word: str) -> str:
    normalized = unicodedata.normalize("NFKD", word.casefold())
    return "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )


def _meaningful_words(text: str) -> set[str]:
    words = {_normalize_word(word) for word in _WORD_PATTERN.findall(text)}
    return {word for word in words if len(word) > 2 and word not in _STOP_WORDS}


def _normalized_text(text: str) -> str:
    return " ".join(_normalize_word(word) for word in _WORD_PATTERN.findall(text))


def _word_matches(left: str, right: str) -> bool:
    if left == right:
        return True
    return len(left) >= 6 and len(right) >= 6 and left[:6] == right[:6]


def _overlap_ratio(candidate: str, evidence: str) -> float:
    candidate_words = _meaningful_words(candidate)
    evidence_words = _meaningful_words(evidence)
    if not candidate_words:
        return 0.0
    matched = sum(
        any(_word_matches(word, evidence_word) for evidence_word in evidence_words)
        for word in candidate_words
    )
    return matched / len(candidate_words)


def _quantities(text: str) -> set[str]:
    normalized_words = {_normalize_word(word) for word in _WORD_PATTERN.findall(text)}
    return set(_NUMBER_PATTERN.findall(text)) | (normalized_words & _NUMBER_WORDS)


def _is_grounded_answer(answer: str, evidence: str, question: str) -> bool:
    answer = answer.strip()
    if len(answer) < 20:
        return False

    normalized_question = _normalized_text(question)
    normalized_answer = _normalized_text(answer)
    for required_phrase in ("en erken", "en gec"):
        if (
            required_phrase in normalized_question
            and required_phrase not in normalized_answer
        ):
            return False

    # New numbers or ordinal words are a strong hallucination signal in regulations.
    if not _quantities(answer).issubset(_quantities(evidence)):
        return False

    # A grounded answer should reuse the important facts and terms in the retrieved
    # text. This rejects fluent-looking but unrelated small-model output.
    return _overlap_ratio(answer, evidence) >= 0.45


def _extractive_fallback(question: str, result: SearchResult) -> str:
    question_words = _meaningful_words(question)
    candidates = [
        sentence.strip(" -#\t")
        for sentence in _SENTENCE_PATTERN.split(result.content)
        if len(sentence.strip(" -#\t")) >= 20
    ]
    if not candidates:
        candidates = [result.content.strip()]

    def sentence_score(sentence: str) -> tuple[float, int]:
        sentence_words = _meaningful_words(sentence)
        if not question_words:
            return (0.0, -len(sentence))
        matched = sum(
            any(_word_matches(word, sentence_word) for sentence_word in sentence_words)
            for word in question_words
        )
        return (matched / len(question_words), -len(sentence))

    best_sentence = max(candidates, key=sentence_score)
    return (
        "Kaynak metinde şu bilgi yer alıyor:\n\n"
        f"{best_sentence}\n\n"
        f"Kaynaklar:\n- {result.citation}"
    )


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Embedding boyutları eşit değil.")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


class RagService:
    def __init__(self, settings: Settings, runtime: object):
        settings.ensure_directories()
        self.settings = settings
        self.runtime = runtime
        self.database = RagDatabase(settings.db_path)

    def index_documents(
        self, rebuild: bool = False, progress: Callable[[str], None] | None = None
    ) -> IndexReport:
        report = progress or (lambda _: None)
        paths = discover_documents(self.settings.data_dir)
        if rebuild:
            self.database.clear()

        indexed_documents = 0
        skipped_documents = 0
        indexed_chunks = 0

        for path in paths:
            relative_path = str(path.relative_to(self.settings.data_dir))
            digest = file_sha256(path)
            if not rebuild and self.database.is_current(relative_path, digest):
                report(f"Atlandı (değişmedi): {relative_path}")
                skipped_documents += 1
                continue

            report(f"Okunuyor: {relative_path}")
            pages = read_document(path)
            chunks = build_chunks(
                pages,
                max_chars=self.settings.chunk_size,
                overlap=self.settings.chunk_overlap,
            )
            if not chunks:
                report(f"Metin bulunamadı: {relative_path}")
                continue

            report(f"Embedding üretiliyor: {relative_path} ({len(chunks)} chunk)")
            embeddings = self.runtime.embed_many([chunk.content for chunk in chunks])
            self.database.replace_document(
                source_path=relative_path,
                source_name=path.name,
                sha256=digest,
                chunks=chunks,
                embeddings=embeddings,
            )
            indexed_documents += 1
            indexed_chunks += len(chunks)

        return IndexReport(
            discovered_documents=len(paths),
            indexed_documents=indexed_documents,
            skipped_documents=skipped_documents,
            indexed_chunks=indexed_chunks,
        )

    def search(self, question: str, top_k: int | None = None) -> list[SearchResult]:
        question = question.strip()
        if not question:
            raise ValueError("Soru boş olamaz.")

        stored_chunks = self.database.all_chunks()
        if not stored_chunks:
            raise RuntimeError("Önce belgeleri indekslemelisiniz.")

        query_embedding = self.runtime.embed_one(question)
        results = [
            SearchResult(
                chunk_id=chunk.chunk_id,
                source_name=chunk.source_name,
                source_path=chunk.source_path,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                content=chunk.content,
                score=cosine_similarity(query_embedding, chunk.embedding),
            )
            for chunk in stored_chunks
        ]
        results.sort(key=lambda item: item.score, reverse=True)
        return results[: top_k or self.settings.top_k]

    def ask(self, question: str) -> RagAnswer:
        results = self.search(question)
        if not results or results[0].score < self.settings.min_similarity:
            return RagAnswer(
                question=question,
                answer=(
                    "Bu sorunun cevabı yüklenen üniversite belgelerinde bulunamadı. "
                    "Güncel ve kesin bilgi için üniversitenin resmî birimine başvurun."
                ),
                sources=results,
                grounded=False,
            )

        # Keep chunks close to the best result. Weak, unrelated chunks distract
        # small local models and can introduce facts from a different regulation.
        context_floor = max(
            self.settings.min_similarity,
            results[0].score - 0.08,
        )
        context_results = [
            result for result in results if result.score >= context_floor
        ]

        context_sections = []
        for index, result in enumerate(context_results, start=1):
            context_sections.append(
                f"[KAYNAK {index}: {result.citation}; chunk {result.chunk_index}; "
                f"benzerlik {result.score:.3f}]\n{result.content}"
            )
        context = "\n\n".join(context_sections)

        messages = [
            {
                "role": "system",
                "content": (
                    "Sen UniGuide adlı üniversite mevzuat asistanısın. Yalnızca "
                    "kullanıcının verdiği kaynak metne dayan. Kaynakta olmayan bilgi, "
                    "sayı veya koşul ekleme. Türkçe ve en fazla üç cümleyle cevap ver."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"KAYNAK METİN:\n{context}\n\n"
                    f"SORU:\n{question.strip()}\n\n"
                    "Sayıları ve yarıyıl bilgilerini kaynakta yazıldığı biçimde koru. "
                    "Sadece cevabı yaz; kaynak listesini uygulama ayrıca ekleyecek."
                ),
            },
        ]
        answer = self.runtime.complete(messages)

        evidence = "\n".join(result.content for result in context_results)
        if not _is_grounded_answer(answer, evidence, question):
            answer = _extractive_fallback(question, context_results[0])
        else:
            citations = "\n".join(
                f"- {result.citation}" for result in context_results
            )
            answer = f"{answer}\n\nKaynaklar:\n{citations}"

        return RagAnswer(
            question=question,
            answer=answer,
            sources=context_results,
            grounded=True,
        )
