from __future__ import annotations

import base64
import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import httpx

from ..catalog import ModelSpec
from ..config import AppConfig, app_home
from ..hardware import find_llama_server
from ..manager import ModelManager
from .base import ChatEngine, ChatOptions, EngineError, EngineUnavailable

logger = logging.getLogger(__name__)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


class LlamaCppEngine(ChatEngine):
    def __init__(self, spec: ModelSpec, config: AppConfig, manager: ModelManager):
        self.spec = spec
        self.model_id = spec.id
        self.config = config
        self.manager = manager
        self._proc: subprocess.Popen | None = None
        self._log_handle = None
        self._log_path: Path | None = None
        self._port: int | None = None
        self._server: str | None = None
        self._model_file: Path | None = None
        self._mmproj: Path | None = None
        self._load_time = 0.0

    def _messages_with_images(self, messages: list[dict], images: list[bytes]) -> list[dict]:
        payload = [
            {"role": message.get("role", "user"), "content": message.get("content", "")}
            for message in messages
        ]
        if images and not self._mmproj:
            raise EngineError(
                "This GGUF build has no mmproj file, so image input is unavailable. "
                "Use a multimodal GGUF with a matching mmproj."
            )
        if images:
            index = next(
                (i for i in range(len(payload) - 1, -1, -1) if payload[i]["role"] == "user"),
                None,
            )
            if index is None:
                payload.append({"role": "user", "content": "Describe the attached image."})
                index = len(payload) - 1
            text = payload[index]["content"] or "Describe the attached image."
            parts: list[dict] = [{"type": "text", "text": text}]
            for image in images:
                parts.append({"type": "image_url", "image_url": {"url": _data_url(image)}})
            payload[index]["content"] = parts
        return payload

    def load(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        if not self.manager.installed(self.spec):
            raise EngineUnavailable(
                f"Model '{self.spec.id}' is not installed. Run: lforge models download {self.spec.id}"
            )
        server = find_llama_server(self.config.llama_server)
        if not server:
            raise EngineUnavailable(
                "llama-server.exe was not found. Install llama.cpp with "
                "'winget install ggml.llamacpp', or set llama_server in the config."
            )
        model_file = self.manager.gguf_file(self.spec)
        if not model_file:
            raise EngineUnavailable(f"No .gguf file found for {self.spec.id}")
        mmproj = self.manager.mmproj_file(self.spec)
        port = _free_port()
        log_dir = app_home() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"llama-{self.spec.id.replace('/', '-')}.log"
        log_handle = open(log_path, "ab", buffering=0)

        args = [
            server,
            "-m",
            str(model_file),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-c",
            str(int(self.config.llama_ctx)),
            "-b",
            str(int(self.config.llama_batch)),
            "-ub",
            str(int(self.config.llama_ubatch)),
            "-ngl",
            str(int(self.config.llama_ngl)),
            "--flash-attn",
            str(self.config.llama_flash_attn),
            "--cache-type-k",
            str(self.config.llama_cache_k),
            "--cache-type-v",
            str(self.config.llama_cache_v),
            "--parallel",
            str(max(1, int(self.config.llama_parallel))),
            "--alias",
            self.model_id,
            "--jinja",
            "--no-webui",
        ]
        if int(self.config.llama_threads) > 0:
            args.extend(["--threads", str(int(self.config.llama_threads))])
        if mmproj:
            args.extend(["--mmproj", str(mmproj)])

        print(
            f"Starting llama.cpp for {self.spec.id} on port {port} "
            f"(ngl={self.config.llama_ngl}, ctx={self.config.llama_ctx})...",
            file=sys.stderr,
            flush=True,
        )
        started = time.perf_counter()
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        deadline = time.time() + 240
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2)
                if response.status_code == 200:
                    ready = True
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        if not ready:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            log_handle.close()
            tail = ""
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
            except OSError:
                pass
            raise EngineUnavailable(
                f"llama-server did not become ready. Log: {log_path}\n{tail}"
            )

        self._proc = proc
        self._log_handle = log_handle
        self._log_path = log_path
        self._port = port
        self._server = server
        self._model_file = model_file
        self._mmproj = mmproj
        self._load_time = time.perf_counter() - started
        print(
            f"Loaded {self.spec.id} with llama.cpp in {self._load_time:.1f}s"
            + (f" (mmproj: {mmproj.name})" if mmproj else " (text-only)"),
            file=sys.stderr,
            flush=True,
        )
        if self.config.warmup:
            self._warmup()

    def _warmup(self) -> None:
        """Absorb the Qwen3.5 hybrid first-request penalty (llama.cpp #29513)."""
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=240) as client:
                client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.model_id,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 8,
                        "temperature": 0,
                        "stream": False,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
            print(
                f"Warmup finished in {time.perf_counter() - started:.1f}s",
                file=sys.stderr,
                flush=True,
            )
        except httpx.HTTPError as exc:
            print(f"Warmup failed (ignored): {exc}", file=sys.stderr, flush=True)

    @property
    def base_url(self) -> str:
        if not self._port:
            raise EngineError("llama.cpp server is not running")
        return f"http://127.0.0.1:{self._port}"

    def _payload(self, messages: list[dict], images: list[bytes], options: ChatOptions, stream: bool) -> dict:
        payload = {
            "model": self.model_id,
            "messages": self._messages_with_images(messages, images),
            "stream": stream,
            "max_tokens": max(1, int(options.max_tokens)),
            "temperature": float(options.temperature),
            "top_p": float(options.top_p),
        }
        if options.seed is not None:
            payload["seed"] = int(options.seed)
        template_kwargs = dict(options.chat_template_kwargs or {})
        if self.config.disable_thinking and "enable_thinking" not in template_kwargs:
            template_kwargs["enable_thinking"] = False
        if template_kwargs:
            payload["chat_template_kwargs"] = template_kwargs
        if options.tools:
            payload["tools"] = options.tools
            if options.tool_choice is not None:
                payload["tool_choice"] = options.tool_choice
        return payload

    def _raw_payload(self, messages: list[dict], options: ChatOptions, stream: bool) -> dict:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": stream,
            "max_tokens": max(1, int(options.max_tokens)),
            "temperature": float(options.temperature),
            "top_p": float(options.top_p),
        }
        if options.seed is not None:
            payload["seed"] = int(options.seed)
        template_kwargs = dict(options.chat_template_kwargs or {})
        if self.config.disable_thinking and "enable_thinking" not in template_kwargs:
            template_kwargs["enable_thinking"] = False
        if template_kwargs:
            payload["chat_template_kwargs"] = template_kwargs
        if options.tools:
            payload["tools"] = options.tools
            if options.tool_choice is not None:
                payload["tool_choice"] = options.tool_choice
        return payload

    def chat_raw(self, messages: list[dict], options: ChatOptions) -> dict:
        """Forward an OpenAI tool-aware request and return llama.cpp's full JSON."""
        self.load()
        try:
            with httpx.Client(timeout=600) as client:
                response = client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=self._raw_payload(messages, options, stream=False),
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:
            raise EngineError(f"llama.cpp tool request failed: {exc}") from exc

    def stream_raw(self, messages: list[dict], options: ChatOptions) -> Iterator[str]:
        """Forward llama.cpp's raw SSE so tool_call deltas reach the client."""
        self.load()
        payload = self._raw_payload(messages, options, stream=True)
        try:
            with httpx.Client(timeout=None) as client:
                with client.stream(
                    "POST", f"{self.base_url}/v1/chat/completions", json=payload
                ) as response:
                    if response.status_code >= 400:
                        body = response.read().decode("utf-8", errors="replace")
                        raise EngineError(f"llama.cpp error {response.status_code}: {body}")
                    for line in response.iter_lines():
                        if not line:
                            continue
                        if isinstance(line, bytes):
                            line = line.decode("utf-8", errors="replace")
                        yield line + "\n\n"
        except httpx.HTTPError as exc:
            raise EngineError(f"llama.cpp tool streaming failed: {exc}") from exc

    def chat(self, messages: list[dict], images: list[bytes], options: ChatOptions) -> str:
        self.load()
        try:
            with httpx.Client(timeout=600) as client:
                response = client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=self._payload(messages, images, options, stream=False),
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise EngineError(f"llama.cpp request failed: {exc}") from exc
        choices = data.get("choices") or []
        if not choices:
            raise EngineError(f"llama.cpp returned no choices: {data}")
        message = choices[0].get("message") or {}
        return (message.get("content") or message.get("reasoning_content") or "").strip()

    def stream_chat(
        self, messages: list[dict], images: list[bytes], options: ChatOptions
    ) -> Iterator[str]:
        self.load()
        payload = self._payload(messages, images, options, stream=True)
        try:
            with httpx.Client(timeout=None) as client:
                with client.stream(
                    "POST", f"{self.base_url}/v1/chat/completions", json=payload
                ) as response:
                    if response.status_code >= 400:
                        body = response.read().decode("utf-8", errors="replace")
                        raise EngineError(f"llama.cpp error {response.status_code}: {body}")
                    for line in response.iter_lines():
                        if not line:
                            continue
                        if isinstance(line, bytes):
                            line = line.decode("utf-8", errors="replace")
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content") or delta.get("reasoning_content")
                        if content:
                            yield content
        except httpx.HTTPError as exc:
            raise EngineError(f"llama.cpp streaming failed: {exc}") from exc

    def unload(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except OSError:
                pass
        self._log_handle = None
        self._port = None
        print(f"Unloaded {self.spec.id}", file=sys.stderr, flush=True)

    def describe(self) -> dict:
        return {
            "id": self.spec.id,
            "engine": "llama.cpp",
            "server": self._server,
            "model_file": str(self._model_file) if self._model_file else None,
            "mmproj_file": str(self._mmproj) if self._mmproj else None,
            "port": self._port,
            "ngl": self.config.llama_ngl,
            "ctx": self.config.llama_ctx,
            "ubatch": self.config.llama_ubatch,
            "parallel": self.config.llama_parallel,
            "kv_cache": f"k={self.config.llama_cache_k},v={self.config.llama_cache_v}",
            "flash_attn": self.config.llama_flash_attn,
            "loaded": self._proc is not None and self._proc.poll() is None,
            "load_time_s": round(self._load_time, 1),
            "modalities": list(self.spec.modalities),
            "log": str(self._log_path) if self._log_path else None,
        }
