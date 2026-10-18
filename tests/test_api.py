from __future__ import annotations

import io
import json

from fastapi.testclient import TestClient
from PIL import Image

from local_forge.api import create_app
from local_forge.catalog import CATALOG, find_spec
from local_forge.config import AppConfig
from local_forge.engines import EngineManager, ImageResult
from local_forge.manager import ModelManager


class FakeChatEngine:
    def __init__(self, model_id):
        self.model_id = model_id

    def load(self):
        return None

    def chat(self, messages, images, options):
        prefix = f"{len(images)} image(s); " if images else ""
        return prefix + "Hello from LocalForge."

    def stream_chat(self, messages, images, options):
        yield "Hello"
        yield " from"
        yield " LocalForge."

    def unload(self):
        return None

    def describe(self):
        return {"id": self.model_id, "engine": "fake", "loaded": True}


class FakeImageEngine:
    def __init__(self):
        self.model_path = "fake"

    def generate(self, prompt, **kwargs):
        return ImageResult(
            images=[Image.new("RGB", (8, 8), (10, 20, 30))],
            seed=42,
            width=256,
            height=256,
        )


class FakeEngineManager:
    def __init__(self):
        self.closed = False

    def get_chat_engine(self, model_id):
        return FakeChatEngine(model_id)

    def get_image_engine(self, model_path=None):
        return FakeImageEngine()

    def status(self):
        return {"active_chat": None, "active_image": None}

    def close(self):
        self.closed = True


class FakeModelManager:
    def __init__(self, config):
        self.config = config

    def specs(self):
        return list(CATALOG)

    def installed(self, spec):
        return spec.id == "gemma4-e2b-uncensored-ov"

    def status(self, spec):
        return {
            "id": spec.id,
            "name": spec.name,
            "engine": spec.engine,
            "installed": self.installed(spec),
            "modalities": list(spec.modalities),
            "license": spec.license,
            "size_gb": spec.size_gb,
            "path": str(self.config.models_path / spec.dir_name),
            "model_file": None,
            "mmproj_file": None,
        }


def _client(tmp_path) -> TestClient:
    config = AppConfig(models_dir=str(tmp_path / "models"))
    config.ensure_dirs()
    models = FakeModelManager(config)
    app = create_app(config, models, FakeEngineManager())
    return TestClient(app)


def test_models_endpoint(tmp_path):
    client = _client(tmp_path)
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == len(CATALOG)
    assert data["data"][0]["installed"] is True


def test_chat_non_stream(tmp_path):
    client = _client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemma4-e2b-uncensored-ov",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["choices"][0]["message"]["content"].startswith("Hello")


def test_chat_stream(tmp_path):
    client = _client(tmp_path)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gemma4-e2b-uncensored-ov",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        text = "".join(response.iter_text())
    assert "Hello" in text
    assert "data: [DONE]" in text


def test_missing_model_returns_409(tmp_path):
    client = _client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-4b-abliterated",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 409


def test_image_generation(tmp_path):
    client = _client(tmp_path)
    response = client.post(
        "/v1/images/generations",
        json={"prompt": "test", "size": "256x256", "response_format": "b64_json"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["data"][0]["b64_json"]
    assert payload["seed"] == 42
