from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any

import httpx
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict

MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_IMAGES = 4


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    model_config = ConfigDict(extra="allow")


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    enable_thinking: bool | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    model_config = ConfigDict(extra="allow")


class ImageGenerationRequest(BaseModel):
    model: str | None = None
    prompt: str
    n: int = 1
    size: str = "512x768"
    response_format: str = "b64_json"
    steps: int | None = None
    guidance_scale: float | None = None
    seed: int | None = None
    negative_prompt: str | None = None
    model_config = ConfigDict(extra="allow")


@dataclass
class PreparedChat:
    messages: list[dict[str, str]]
    images: list[bytes]
    had_image_parts: bool


def _decode_data_url(url: str) -> bytes:
    if "," not in url or not url.startswith("data:"):
        raise ValueError("Invalid image data URL")
    header, encoded = url.split(",", 1)
    if ";base64" not in header:
        raise ValueError("Only base64 image data URLs are supported")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError(f"Invalid base64 image data: {exc}") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the 15 MB limit")
    return raw


def _fetch_image(url: str) -> bytes:
    if url.startswith("data:"):
        return _decode_data_url(url)
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("Only http(s) URLs and base64 data URLs are supported for images")
    try:
        response = httpx.get(url, follow_redirects=True, timeout=30)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ValueError(f"Could not download image: {exc}") from exc
    if len(response.content) > MAX_IMAGE_BYTES:
        raise ValueError("Downloaded image exceeds the 15 MB limit")
    return response.content


def _normalize_png(data: bytes) -> bytes:
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Unsupported image: {exc}") from exc
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def load_image_file(path: str) -> bytes:
    from pathlib import Path

    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Image not found: {source}")
    data = source.read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the 15 MB limit")
    return _normalize_png(data)


def prepare_messages(messages: list[ChatMessage]) -> PreparedChat:
    clean: list[dict[str, str]] = []
    images: list[bytes] = []
    had_image_parts = False
    for message in messages:
        role = message.role.strip().lower()
        if role == "developer":
            role = "system"
        content = message.content
        text_parts: list[str] = []
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = str(part.get("type", ""))
                if part_type in {"text", "input_text"}:
                    value = part.get("text") or part.get("value")
                    if isinstance(value, dict):
                        value = value.get("value")
                    if value:
                        text_parts.append(str(value))
                elif part_type in {"image_url", "input_image"}:
                    had_image_parts = True
                    image_url = part.get("image_url") or part.get("url")
                    if isinstance(image_url, dict):
                        image_url = image_url.get("url")
                    if not image_url:
                        raise ValueError("Image part is missing a URL")
                    if len(images) >= MAX_IMAGES:
                        raise ValueError(f"At most {MAX_IMAGES} images are supported per request")
                    images.append(_normalize_png(_fetch_image(str(image_url))))
        text = "\n".join(part for part in text_parts if part).strip()
        if not text and had_image_parts and role == "user":
            text = "Describe the attached image."
        clean.append({"role": role, "content": text})
    if not clean:
        raise ValueError("At least one message is required")
    return PreparedChat(messages=clean, images=images, had_image_parts=had_image_parts)
