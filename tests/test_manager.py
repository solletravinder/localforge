from __future__ import annotations

from local_forge.catalog import CATALOG
from local_forge.config import AppConfig
from local_forge.manager import ModelManager


def test_gguf_local_model(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALFORGE_HOME", str(tmp_path / "home"))
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"GGUF" + b"0" * 64)
    config = AppConfig(models_dir=str(tmp_path / "models"))
    manager = ModelManager(config)
    spec = manager.add_local("test-gguf", str(model_file), "llama.cpp", name="Test GGUF")
    assert manager.installed(spec)
    assert manager.gguf_file(spec) == model_file
    assert manager.mmproj_file(spec) is None
    manager.remove_local("test-gguf")
    assert all(item.id != "test-gguf" for item in manager.specs())


def test_openvino_local_model(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALFORGE_HOME", str(tmp_path / "home"))
    model_dir = tmp_path / "ov-model"
    model_dir.mkdir()
    (model_dir / "openvino_language_model.xml").write_text("<xml/>", encoding="utf-8")
    config = AppConfig(models_dir=str(tmp_path / "models"))
    manager = ModelManager(config)
    spec = manager.add_local("test-ov", str(model_dir), "openvino", modalities=["text", "image"])
    assert manager.installed(spec)
    assert spec.modalities == ("text", "image")


def test_litert_local_model(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALFORGE_HOME", str(tmp_path / "home"))
    model_file = tmp_path / "model.litertlm"
    model_file.write_bytes(b"LITERTLM" + b"0" * 64)
    config = AppConfig(models_dir=str(tmp_path / "models"))
    manager = ModelManager(config)
    spec = manager.add_local(
        "test-litert", str(model_file), "litert", name="Test LiteRT", modalities=["text"]
    )
    assert manager.installed(spec)
    assert spec.format == "litertlm"
    assert manager.path_for(spec) == model_file


def test_catalog_defaults_are_uncensored():
    assert all(spec.engine in {"openvino", "llama.cpp"} for spec in CATALOG)
    assert any(spec.id.startswith("gemma4") for spec in CATALOG)
