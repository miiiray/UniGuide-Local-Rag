from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


ProgressCallback = Callable[[str], None]


class FoundryLocalRuntime:
    """Lazy wrapper around the current Microsoft Foundry Local native SDK."""

    def __init__(
        self,
        embedding_model: str,
        chat_model: str,
        progress: ProgressCallback | None = None,
    ):
        self.embedding_model_name = embedding_model
        self.chat_model_name = chat_model
        self.progress = progress or (lambda _: None)
        self._manager: Any = None
        self._embedding_model: Any = None
        self._chat_model: Any = None
        self._embedding_client: Any = None
        self._chat_client: Any = None

    def _initialize_manager(self) -> None:
        if self._manager is not None:
            return

        try:
            from foundry_local_sdk import Configuration, FoundryLocalManager
        except ImportError as exc:
            raise RuntimeError(
                "Foundry Local SDK bulunamadı. Windows'ta "
                "'py -m pip install -r requirements-windows.txt' komutunu çalıştırın."
            ) from exc

        self.progress("Foundry Local başlatılıyor...")
        configuration = Configuration(app_name="uniguide_local_rag")
        FoundryLocalManager.initialize(configuration)
        self._manager = FoundryLocalManager.instance

    def _start_embedding(self) -> None:
        if self._embedding_client is not None:
            return

        self._initialize_manager()
        self._unload_chat()

        if self._embedding_model is None:
            self._embedding_model = self._manager.catalog.get_model(
                self.embedding_model_name
            )
        self.progress(f"Embedding modeli hazırlanıyor: {self.embedding_model_name}")
        self._embedding_model.download(
            lambda value: self.progress(f"Embedding modeli indiriliyor: %{value:.1f}")
        )
        self._embedding_model.load()
        self._embedding_client = self._embedding_model.get_embedding_client()

    def _start_chat(self) -> None:
        if self._chat_client is not None:
            return

        self._initialize_manager()
        self._unload_embedding()

        if self._chat_model is None:
            self._chat_model = self._manager.catalog.get_model(self.chat_model_name)
        self.progress(f"Sohbet modeli hazırlanıyor: {self.chat_model_name}")
        self._chat_model.download(
            lambda value: self.progress(f"Sohbet modeli indiriliyor: %{value:.1f}")
        )
        self._chat_model.load()
        self._chat_client = self._chat_model.get_chat_client()
        self.progress("Sohbet modeli hazır.")

    def _unload_embedding(self) -> None:
        if self._embedding_client is not None and self._embedding_model is not None:
            self._embedding_model.unload()
        self._embedding_client = None

    def _unload_chat(self) -> None:
        if self._chat_client is not None and self._chat_model is not None:
            self._chat_model.unload()
        self._chat_client = None

    def embed_many(self, texts: Sequence[str], batch_size: int = 16) -> list[list[float]]:
        self._start_embedding()
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            response = self._embedding_client.generate_embeddings(batch)
            embeddings.extend([list(item.embedding) for item in response.data])
        return embeddings

    def embed_one(self, text: str) -> list[float]:
        self._start_embedding()
        response = self._embedding_client.generate_embedding(text)
        return list(response.data[0].embedding)

    def complete(self, messages: list[dict[str, str]]) -> str:
        self._start_chat()
        response = self._chat_client.complete_chat(messages)
        return (response.choices[0].message.content or "").strip()

    def close(self) -> None:
        self._unload_chat()
        self._unload_embedding()
