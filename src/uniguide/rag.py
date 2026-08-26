from __future__ import annotations

import math
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

        context_sections = []
        for index, result in enumerate(results, start=1):
            context_sections.append(
                f"[KAYNAK {index}: {result.citation}; chunk {result.chunk_index}; "
                f"benzerlik {result.score:.3f}]\n{result.content}"
            )
        context = "\n\n".join(context_sections)

        messages = [
            {
                "role": "system",
                "content": (
                    "Sen UniGuide adlı üniversite mevzuat asistanısın. Yanıtını yalnızca "
                    "aşağıdaki BAĞLAM içindeki bilgilere dayandır. Bağlam yeterli değilse "
                    "açıkça 'Bu bilgi sağlanan belgelerde bulunmuyor' de. Tahmin etme ve "
                    "genel bilgini kullanma. Kısa, açık ve Türkçe yanıt ver. İlgili madde "
                    "veya koşul bağlamda bulunuyorsa belirt. Yanıtın sonunda kullandığın "
                    "kaynakları 'Kaynaklar:' başlığı altında listele. Bu sistem resmî "
                    "akademik danışmanlık yerine geçmez.\n\n"
                    f"BAĞLAM:\n{context}"
                ),
            },
            {"role": "user", "content": question.strip()},
        ]
        answer = self.runtime.complete(messages)
        return RagAnswer(
            question=question,
            answer=answer,
            sources=results,
            grounded=True,
        )
