from __future__ import annotations

import gc
import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import AppConfig
from .base import EngineError, EngineUnavailable


@dataclass
class ImageResult:
    images: list[Image.Image]
    seed: int
    width: int
    height: int


class ImageEngine:
    def __init__(self, model_path: Path, config: AppConfig):
        self.model_path = model_path
        self.config = config
        self._pipe = None
        self._lock = threading.Lock()
        self._device_requested = config.openvino_device or "CPU"
        self._device = self._device_requested
        self._load_time = 0.0

    def load(self) -> None:
        if self._pipe is not None:
            return
        try:
            import openvino_genai as ovg
        except ImportError as exc:
            raise EngineUnavailable(
                "openvino-genai is not installed. Run: pip install openvino-genai"
            ) from exc
        if not self.model_path.exists():
            raise EngineUnavailable(
                f"Image model path does not exist: {self.model_path}. "
                "Set image_model in config or pass --model-path."
            )
        print(
            f"Loading image model {self.model_path} on {self._device_requested} ...",
            file=sys.stderr,
            flush=True,
        )
        started = time.perf_counter()
        try:
            self._pipe = ovg.Text2ImagePipeline(str(self.model_path), self._device_requested)
        except Exception as exc:
            if self._device_requested.upper() != "CPU":
                print(
                    f"Image model {self._device_requested} failed ({exc}); falling back to CPU.",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    self._pipe = ovg.Text2ImagePipeline(str(self.model_path), "CPU")
                    self._device = "CPU"
                except Exception as cpu_exc:
                    raise EngineUnavailable(f"Could not load image model: {cpu_exc}") from cpu_exc
            else:
                raise EngineUnavailable(f"Could not load image model: {exc}") from exc
        self._load_time = time.perf_counter() - started
        print(
            f"Loaded image model on {self._device} in {self._load_time:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    @staticmethod
    def _to_pil(tensor, index: int) -> Image.Image:
        array = np.array(tensor.data)
        if array.ndim == 4:
            array = array[index]
        if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
        if array.ndim == 3 and array.shape[-1] == 4:
            array = array[..., :3]
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        if array.dtype != np.uint8:
            if float(array.max()) <= 1.5:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array, mode="RGB")

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        guidance_scale: float = 7.0,
        seed: int | None = None,
        num_images: int = 1,
    ) -> ImageResult:
        self.load()
        width = int(width)
        height = int(height)
        if width % 8 or height % 8:
            raise ValueError("width and height must be multiples of 8")
        if not (128 <= width <= 1024 and 128 <= height <= 1024):
            raise ValueError("width and height must be between 128 and 1024")
        steps = max(1, min(int(steps), 100))
        num_images = max(1, min(int(num_images), 4))
        seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)
        if not negative_prompt:
            negative_prompt = self.config.image_negative
        kwargs = {
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": float(guidance_scale),
            "rng_seed": seed,
            "num_images_per_prompt": num_images,
        }
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        try:
            with self._lock:
                tensor = self._pipe.generate(prompt, **kwargs)
        except Exception as exc:
            raise EngineError(f"Image generation failed: {exc}") from exc
        images = [self._to_pil(tensor, index) for index in range(num_images)]
        return ImageResult(images=images, seed=seed, width=width, height=height)

    def unload(self) -> None:
        with self._lock:
            self._pipe = None
            gc.collect()
        print("Unloaded image model", file=sys.stderr, flush=True)

    def describe(self) -> dict:
        return {
            "engine": "openvino-text2image",
            "model_path": str(self.model_path),
            "device_requested": self._device_requested,
            "device": self._device,
            "loaded": self._pipe is not None,
            "load_time_s": round(self._load_time, 1),
        }
