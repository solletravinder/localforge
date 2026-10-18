from __future__ import annotations

import gc
import io
import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from ..catalog import ModelSpec
from ..config import AppConfig
from ..manager import ModelManager
from ..tools import (
    ToolChatResult,
    parse_gemma_tool_calls,
    sanitize_tools_for_gemma,
)
from .base import ChatEngine, ChatOptions, EngineError, EngineUnavailable

logger = logging.getLogger(__name__)


class OpenVINOEngine(ChatEngine):
    def __init__(self, spec: ModelSpec, config: AppConfig, manager: ModelManager):
        self.spec = spec
        self.model_id = spec.id
        self.config = config
        self.manager = manager
        self._pipe = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._device_requested = config.openvino_device or "CPU"
        self._device = self._device_requested
        self._load_time = 0.0

    @property
    def model_path(self) -> Path:
        return self.manager.path_for(self.spec)

    def load(self) -> None:
        if self._pipe is not None:
            return
        try:
            import openvino_genai as ovg
        except ImportError as exc:
            raise EngineUnavailable(
                "openvino-genai is not installed. Run: pip install openvino-genai"
            ) from exc
        if not self.manager.installed(self.spec):
            raise EngineUnavailable(
                f"Model '{self.spec.id}' is not installed. Run: lforge models download {self.spec.id}"
            )
        path = str(self.model_path)
        print(
            f"Loading {self.spec.id} with OpenVINO on {self._device_requested} "
            "(first load can take 1-2 minutes)...",
            file=sys.stderr,
            flush=True,
        )
        started = time.perf_counter()
        try:
            self._pipe = ovg.VLMPipeline(path, self._device_requested)
        except Exception as exc:
            if self._device_requested.upper() != "CPU":
                print(
                    f"OpenVINO {self._device_requested} failed ({exc}); falling back to CPU.",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    self._pipe = ovg.VLMPipeline(path, "CPU")
                    self._device = "CPU"
                except Exception as cpu_exc:
                    raise EngineUnavailable(
                        f"OpenVINO could not load {self.spec.id}: {cpu_exc}"
                    ) from cpu_exc
            else:
                raise EngineUnavailable(
                    f"OpenVINO could not load {self.spec.id}: {exc}"
                ) from exc
        self._tokenizer = self._pipe.get_tokenizer()
        self._load_time = time.perf_counter() - started
        print(
            f"Loaded {self.spec.id} on {self._device} in {self._load_time:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    def _prompt(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        assert self._tokenizer is not None
        clean = []
        for message in messages:
            item = {
                "role": message.get("role", "user"),
                "content": message.get("content", ""),
            }
            for key in ("tool_calls", "tool_responses"):
                if message.get(key):
                    item[key] = message[key]
            clean.append(item)
        try:
            if tools:
                return self._tokenizer.apply_chat_template(
                    clean, add_generation_prompt=True, tools=tools
                )
            return self._tokenizer.apply_chat_template(clean, add_generation_prompt=True)
        except Exception as exc:
            logger.warning("Chat template failed (%s); using fallback template", exc)
            parts = [
                f"<|{message['role']}|>\n{message['content']}\n" for message in clean
            ]
            return "".join(parts) + "<|assistant|>\n"

    def _tensors(self, images: list[bytes]):
        import openvino as ov

        tensors = []
        for data in images:
            image = Image.open(io.BytesIO(data)).convert("RGB")
            if max(image.size) > 1536:
                image.thumbnail((1536, 1536))
            array = np.asarray(image, dtype=np.uint8)
            tensors.append(ov.Tensor(array[None, ...]))
        return tensors

    @staticmethod
    def _generation_kwargs(options: ChatOptions) -> dict:
        kwargs: dict = {
            "max_new_tokens": max(1, int(options.max_tokens)),
            "do_sample": options.temperature > 0,
        }
        if options.temperature > 0:
            kwargs["temperature"] = float(options.temperature)
            kwargs["top_p"] = float(options.top_p)
            if options.top_k and options.top_k > 0:
                kwargs["top_k"] = int(options.top_k)
        if options.repetition_penalty and options.repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = float(options.repetition_penalty)
        return kwargs

    def chat(self, messages: list[dict], images: list[bytes], options: ChatOptions) -> str:
        self.load()
        prompt = self._prompt(messages)
        tensors = self._tensors(images) if images else None
        kwargs = self._generation_kwargs(options)
        try:
            with self._lock:
                if tensors:
                    result = self._pipe.generate(prompt, images=tensors, **kwargs)
                else:
                    result = self._pipe.generate(prompt, **kwargs)
        except Exception as exc:
            raise EngineError(f"OpenVINO generation failed: {exc}") from exc
        return result.texts[0].strip()

    def chat_tools(
        self,
        messages: list[dict],
        images: list[bytes],
        options: ChatOptions,
        tools: list[dict] | None = None,
    ) -> ToolChatResult:
        self.load()
        selected_tools = sanitize_tools_for_gemma(tools or options.tools)
        prompt = self._prompt(messages, tools=selected_tools)
        tensors = self._tensors(images) if images else None
        kwargs = self._generation_kwargs(options)
        try:
            with self._lock:
                if tensors:
                    result = self._pipe.generate(prompt, images=tensors, **kwargs)
                else:
                    result = self._pipe.generate(prompt, **kwargs)
        except Exception as exc:
            raise EngineError(f"OpenVINO tool generation failed: {exc}") from exc
        raw_text = result.texts[0].strip()
        fallback = next(
            (
                str(message.get("content") or "").strip()
                for message in messages
                if message.get("role") == "user"
            ),
            "",
        )
        calls, content = parse_gemma_tool_calls(
            raw_text, selected_tools, fallback_query=fallback or None
        )
        return ToolChatResult(content=content, tool_calls=calls, raw_text=raw_text)

    def stream_chat(
        self, messages: list[dict], images: list[bytes], options: ChatOptions
    ) -> Iterator[str]:
        self.load()
        prompt = self._prompt(messages)
        tensors = self._tensors(images) if images else None
        kwargs = self._generation_kwargs(options)
        events: queue.Queue = queue.Queue()

        def streamer(text: str) -> bool:
            if text:
                events.put(("token", text))
            return False

        def run() -> None:
            try:
                with self._lock:
                    if tensors:
                        self._pipe.generate(prompt, images=tensors, streamer=streamer, **kwargs)
                    else:
                        self._pipe.generate(prompt, streamer=streamer, **kwargs)
                events.put(("done", None))
            except Exception as exc:
                events.put(("error", exc))

        thread = threading.Thread(target=run, name=f"ov-{self.model_id}", daemon=True)
        thread.start()
        while True:
            kind, payload = events.get()
            if kind == "token":
                yield payload
            elif kind == "error":
                raise EngineError(f"OpenVINO streaming failed: {payload}") from payload
            else:
                break

    def unload(self) -> None:
        with self._lock:
            self._pipe = None
            self._tokenizer = None
            gc.collect()
        print(f"Unloaded {self.spec.id}", file=sys.stderr, flush=True)

    def describe(self) -> dict:
        return {
            "id": self.spec.id,
            "engine": "openvino",
            "model_path": str(self.model_path),
            "device_requested": self._device_requested,
            "device": self._device,
            "loaded": self._pipe is not None,
            "load_time_s": round(self._load_time, 1),
            "modalities": list(self.spec.modalities),
        }
