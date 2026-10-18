from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


class EngineError(RuntimeError):
    """A recoverable engine failure reported to CLI/API clients."""


class EngineUnavailable(EngineError):
    """The engine or model cannot be loaded on this machine."""


@dataclass
class ChatOptions:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.0
    seed: int | None = None
    chat_template_kwargs: dict | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None


class ChatEngine(ABC):
    model_id: str

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def chat(self, messages: list[dict], images: list[bytes], options: ChatOptions) -> str:
        ...

    @abstractmethod
    def stream_chat(
        self, messages: list[dict], images: list[bytes], options: ChatOptions
    ) -> Iterator[str]:
        ...

    @abstractmethod
    def unload(self) -> None:
        ...

    @abstractmethod
    def describe(self) -> dict:
        ...
