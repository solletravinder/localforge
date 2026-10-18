from __future__ import annotations

import threading

from ..catalog import ModelSpec, find_spec
from ..config import AppConfig
from ..manager import ModelManager
from .base import ChatEngine, EngineUnavailable
from .image_engine import ImageEngine
from .llama_cpp_engine import LlamaCppEngine
from .litert_engine import LiteRTEngine
from .openvino_engine import OpenVINOEngine


class EngineManager:
    """Keeps exactly one text engine and one image engine resident.

    Switching chat models unloads the previous model so a 16 GB machine does not
    hold an OpenVINO pipeline and a llama.cpp server at the same time.
    """

    def __init__(self, config: AppConfig, manager: ModelManager):
        self.config = config
        self.manager = manager
        self._chat: ChatEngine | None = None
        self._image: ImageEngine | None = None
        self._lock = threading.RLock()

    @property
    def active_model_id(self) -> str | None:
        return self._chat.model_id if self._chat else None

    def get_chat_engine(self, model_id: str | None) -> ChatEngine:
        spec = find_spec(self.config, model_id)
        with self._lock:
            if self._chat is not None and self._chat.model_id == spec.id:
                return self._chat
            self.close_chat()
            self.close_image()
            if spec.engine == "openvino":
                engine: ChatEngine = OpenVINOEngine(spec, self.config, self.manager)
            elif spec.engine == "llama.cpp":
                engine = LlamaCppEngine(spec, self.config, self.manager)
            elif spec.engine == "litert":
                engine = LiteRTEngine(spec, self.config, self.manager)
            else:
                raise EngineUnavailable(f"Unsupported engine '{spec.engine}' for {spec.id}")
            engine.load()
            self._chat = engine
            return engine

    def get_image_engine(self, model_path: str | None = None) -> ImageEngine:
        selected = model_path or self.config.image_model
        if not selected:
            raise EngineUnavailable(
                "No image model configured. Export a Stable Diffusion model to OpenVINO IR and "
                "set image_model, or pass --model-path. See: lforge convert-image --help"
            )
        from pathlib import Path

        path = Path(selected).expanduser()
        with self._lock:
            if self._image is not None and self._image.model_path == path and self._image._pipe is not None:
                return self._image
            self.close_chat()
            self.close_image()
            engine = ImageEngine(path, self.config)
            engine.load()
            self._image = engine
            return engine

    def close_chat(self) -> None:
        with self._lock:
            if self._chat is not None:
                try:
                    self._chat.unload()
                finally:
                    self._chat = None

    def close_image(self) -> None:
        with self._lock:
            if self._image is not None:
                try:
                    self._image.unload()
                finally:
                    self._image = None

    def close(self) -> None:
        self.close_chat()
        self.close_image()

    def status(self) -> dict:
        return {
            "active_chat": self._chat.describe() if self._chat else None,
            "active_image": self._image.describe() if self._image else None,
        }
