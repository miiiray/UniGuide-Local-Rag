import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from uniguide.foundry import FoundryLocalRuntime


def completion_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


class FoundryRuntimeTests(unittest.TestCase):
    def test_chat_cancellation_reloads_model_and_retries_once(self) -> None:
        runtime = FoundryLocalRuntime("embedding", "chat")
        first_client = Mock()
        first_client.complete_chat.side_effect = RuntimeError(
            "Error during chat completion: Operation was canceled"
        )
        second_client = Mock()
        second_client.complete_chat.return_value = completion_response("Yanıt")
        clients = iter([first_client, second_client])

        def start_chat() -> None:
            if runtime._chat_client is None:
                runtime._chat_client = next(clients)

        runtime._start_chat = start_chat  # type: ignore[method-assign]
        runtime._unload_chat = lambda: setattr(  # type: ignore[method-assign]
            runtime, "_chat_client", None
        )

        answer = runtime.complete([{"role": "user", "content": "Soru"}])

        self.assertEqual(answer, "Yanıt")
        first_client.complete_chat.assert_called_once()
        second_client.complete_chat.assert_called_once()

    def test_non_transient_chat_error_is_not_retried(self) -> None:
        runtime = FoundryLocalRuntime("embedding", "chat")
        client = Mock()
        client.complete_chat.side_effect = RuntimeError("Model yüklenemedi")
        runtime._chat_client = client

        with self.assertRaisesRegex(RuntimeError, "Model yüklenemedi"):
            runtime.complete([{"role": "user", "content": "Soru"}])

        client.complete_chat.assert_called_once()


if __name__ == "__main__":
    unittest.main()
