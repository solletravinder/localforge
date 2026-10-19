from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

APP_NAME = "LocalForge"

DISCLAIMER = (
    "LocalForge runs uncensored/abliterated community models with no content filter. "
    "You are responsible for the prompts you send, the content you generate, and compliance "
    "with the laws that apply to you. Model licenses and quality vary; verify both."
)


def app_home() -> Path:
    override = os.environ.get("LOCALFORGE_HOME")
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
        return base / APP_NAME
    return Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")) / "localforge"


def default_models_dir() -> Path:
    override = os.environ.get("LOCALFORGE_MODELS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
        return base / APP_NAME / "models"
    return Path.home() / ".local" / "share" / "localforge" / "models"


def config_path() -> Path:
    override = os.environ.get("LOCALFORGE_CONFIG")
    if override:
        return Path(override).expanduser().resolve()
    return app_home() / "config.json"


@dataclass
class AppConfig:
    models_dir: str = field(default_factory=lambda: str(default_models_dir()))
    openvino_device: str = "GPU"
    llama_server: str = ""
    llama_ngl: int = 10
    llama_ctx: int = 131072
    llama_batch: int = 2048
    llama_ubatch: int = 1024
    llama_threads: int = 0
    llama_parallel: int = 1
    llama_flash_attn: str = "on"
    llama_cache_k: str = "q8_0"
    llama_cache_v: str = "q4_0"
    api_host: str = "127.0.0.1"
    api_port: int = 8090
    image_model: str = ""
    image_width: int = 512
    image_height: int = 768
    image_steps: int = 28
    image_guidance: float = 6.5
    image_negative: str = (
        "child, minor, teen, cartoon, anime, 3d, cgi, painting, illustration, "
        "plastic skin, airbrushed, bad anatomy, bad hands, extra fingers, deformed, "
        "blurry, lowres, watermark, text"
    )
    system_prompt: str = "You are a helpful local assistant. Follow the user's instructions."
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    warmup: bool = True
    disable_thinking: bool = True
    litert_backend: str = "GPU"
    litert_ctx: int = 4096
    litert_validate_tools: bool = True
    first_run_notice: bool = True
    local_models: list[dict] = field(default_factory=list)

    @property
    def models_path(self) -> Path:
        return Path(self.models_dir).expanduser()

    @property
    def image_model_path(self) -> Path | None:
        if not self.image_model:
            return None
        return Path(self.image_model).expanduser()

    def ensure_dirs(self) -> None:
        self.models_path.mkdir(parents=True, exist_ok=True)
        app_home().mkdir(parents=True, exist_ok=True)
        (app_home() / "logs").mkdir(parents=True, exist_ok=True)


def load_config() -> AppConfig:
    path = config_path()
    cfg = AppConfig()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read config {path}: {exc}") from exc
        allowed = {f.name for f in fields(AppConfig)}
        cfg = AppConfig(**{k: v for k, v in data.items() if k in allowed})
    models_override = os.environ.get("LOCALFORGE_MODELS_DIR")
    if models_override:
        cfg.models_dir = str(Path(models_override).expanduser())
    cfg.ensure_dirs()
    return cfg


def save_config(cfg: AppConfig) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    return path


def set_config_value(cfg: AppConfig, key: str, raw_value: str) -> None:
    allowed = {f.name: f for f in fields(AppConfig)}
    if key not in allowed:
        raise KeyError(f"Unknown config key: {key}. Valid keys: {', '.join(sorted(allowed))}")
    current = getattr(cfg, key)
    if key == "local_models":
        value = json.loads(raw_value)
    elif isinstance(current, bool):
        value = raw_value.strip().lower() in {"1", "true", "yes", "on"}
    elif isinstance(current, int):
        value = int(raw_value)
    elif isinstance(current, float):
        value = float(raw_value)
    else:
        value = raw_value
    setattr(cfg, key, value)
    save_config(cfg)


def mark_notice_seen(cfg: AppConfig) -> None:
    if cfg.first_run_notice:
        cfg.first_run_notice = False
        save_config(cfg)
