from __future__ import annotations

import argparse
import sys

from uniguide.config import Settings
from uniguide.foundry import FoundryLocalRuntime
from uniguide.rag import RagService


def build_service(settings: Settings) -> RagService:
    runtime = FoundryLocalRuntime(
        embedding_model=settings.embedding_model,
        chat_model=settings.chat_model,
        progress=lambda message: print(message, flush=True),
    )
    return RagService(settings=settings, runtime=runtime)


def print_answer(service: RagService, question: str) -> None:
    result = service.ask(question)
    print(f"\n{result.answer}\n")
    print("Getirilen parçalar:")
    for source in result.sources:
        print(f"- {source.citation} | skor: {source.score:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UniGuide Local RAG - yerel üniversite mevzuat asistanı"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Belgeleri indeksle")
    index_parser.add_argument("--rebuild", action="store_true", help="İndeksi sıfırdan kur")

    ask_parser = subparsers.add_parser("ask", help="Tek bir soru sor")
    ask_parser.add_argument("question")

    subparsers.add_parser("chat", help="Etkileşimli soru-cevap oturumu")
    subparsers.add_parser("status", help="İndeks durumunu göster")

    args = parser.parse_args()
    settings = Settings()
    service = build_service(settings)

    try:
        if args.command == "index":
            report = service.index_documents(rebuild=args.rebuild, progress=print)
            print(
                f"\nTamamlandı: {report.indexed_documents} belge ve "
                f"{report.indexed_chunks} chunk indekslendi; "
                f"{report.skipped_documents} belge değişmediği için atlandı."
            )
        elif args.command == "ask":
            print_answer(service, args.question)
        elif args.command == "status":
            documents, chunks = service.database.stats()
            print(f"İndeks: {documents} belge, {chunks} chunk")
        elif args.command == "chat":
            print("UniGuide hazır. Çıkmak için 'çık' yazın.")
            while True:
                question = input("\nSorunuz: ").strip()
                if question.lower() in {"çık", "cik", "quit", "exit"}:
                    break
                if not question:
                    continue
                print_answer(service, question)
    except (RuntimeError, ValueError) as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        close = getattr(service.runtime, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
