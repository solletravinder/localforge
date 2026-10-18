from __future__ import annotations

import shutil
from pathlib import Path

from .catalog import ImageModelSpec, ModelSpec, all_specs, find_spec, local_specs
from .config import AppConfig, save_config
from .hardware import directory_size


class ModelManager:
    def __init__(self, config: AppConfig):
        self.config = config

    def specs(self) -> list[ModelSpec]:
        return all_specs(self.config)

    def get(self, model_id: str | None) -> ModelSpec:
        return find_spec(self.config, model_id)

    def path_for(self, spec: ModelSpec) -> Path:
        if spec.local_path:
            return Path(spec.local_path).expanduser()
        return self.config.models_path / spec.dir_name

    def installed(self, spec: ModelSpec) -> bool:
        path = self.path_for(spec)
        if spec.format == "openvino":
            return (path / "openvino_language_model.xml").exists()
        if spec.format == "litertlm":
            return path.is_file() and path.suffix.lower() == ".litertlm"
        if path.is_file():
            return path.suffix.lower() == ".gguf"
        return self.gguf_file(spec) is not None

    def gguf_file(self, spec: ModelSpec) -> Path | None:
        path = self.path_for(spec)
        if path.is_file() and path.suffix.lower() == ".gguf":
            return path
        if not path.is_dir():
            return None
        files = [
            item
            for item in path.glob("*.gguf")
            if "mmproj" not in item.name.lower()
        ]
        if not files:
            return None
        preferred = [item for item in files if spec.quant.lower() in item.name.lower()]
        candidates = preferred or files
        return max(candidates, key=lambda item: item.stat().st_size)

    def mmproj_file(self, spec: ModelSpec) -> Path | None:
        path = self.path_for(spec)
        if not path.is_dir():
            return None
        for item in path.glob("*.gguf"):
            if "mmproj" in item.name.lower():
                return item
        return None

    def size_gb(self, spec: ModelSpec) -> float:
        return round(directory_size(self.path_for(spec)), 2)

    def status(self, spec: ModelSpec) -> dict:
        path = self.path_for(spec)
        gguf = self.gguf_file(spec) if spec.format == "gguf" else None
        mmproj = self.mmproj_file(spec) if spec.format == "gguf" else None
        return {
            "id": spec.id,
            "name": spec.name,
            "engine": spec.engine,
            "format": spec.format,
            "repo": spec.repo,
            "license": spec.license,
            "modalities": list(spec.modalities),
            "notes": spec.notes,
            "installed": self.installed(spec),
            "path": str(path),
            "size_gb": self.size_gb(spec),
            "expected_size_gb": spec.size_gb,
            "model_file": str(gguf) if gguf else None,
            "mmproj_file": str(mmproj) if mmproj else None,
            "quant": spec.quant,
            "local": spec.is_local,
        }

    def download(self, spec: ModelSpec, force: bool = False) -> Path:
        if spec.is_local:
            raise RuntimeError(f"'{spec.id}' is a local model. It is not downloadable.")
        if self.installed(spec) and not force:
            print(f"{spec.id} is already installed at {self.path_for(spec)}")
            return self.path_for(spec)
        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is not installed. Run: pip install huggingface_hub") from exc

        target = self.path_for(spec)
        target.mkdir(parents=True, exist_ok=True)
        patterns = list(spec.allow_patterns) or None
        print(f"Downloading {spec.name} ({spec.size_gb:.2f} GB) from {spec.repo} ...")
        try:
            snapshot_download(
                repo_id=spec.repo,
                local_dir=str(target),
                allow_patterns=patterns,
            )
        except Exception as exc:
            hint = ""
            try:
                files = HfApi().list_repo_files(spec.repo)
                ggufs = [name for name in files if name.lower().endswith(".gguf")]
                hint = "\nAvailable GGUF files in the repo: " + ", ".join(ggufs[:20])
            except Exception:
                pass
            raise RuntimeError(f"Download failed for {spec.repo}: {exc}{hint}") from exc

        if not self.installed(spec):
            raise RuntimeError(
                f"Download finished but no usable model was found in {target}. "
                "Check the repository file names or use 'lforge models add-local'."
            )
        print(f"Installed {spec.id} at {target} ({self.size_gb(spec):.2f} GB)")
        return target

    def add_local(
        self,
        model_id: str,
        path: str,
        engine: str,
        name: str | None = None,
        modalities: list[str] | None = None,
    ) -> ModelSpec:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Path does not exist: {resolved}")
        if engine not in {"llama.cpp", "openvino", "litert"}:
            raise ValueError("engine must be 'llama.cpp', 'openvino', or 'litert'")
        if engine == "litert":
            if not resolved.is_file() or resolved.suffix.lower() != ".litertlm":
                raise ValueError("LiteRT models must be a local .litertlm file")
            fmt = "litertlm"
        elif engine == "openvino":
            if not resolved.is_dir() or not (resolved / "openvino_language_model.xml").exists():
                raise ValueError(
                    "OpenVINO models must be a directory containing openvino_language_model.xml"
                )
            fmt = "openvino"
        else:
            if resolved.is_file():
                if resolved.suffix.lower() != ".gguf":
                    raise ValueError("llama.cpp models must be a .gguf file or a directory containing one")
            elif not any(resolved.glob("*.gguf")):
                raise ValueError("No .gguf files found in the selected directory")
            fmt = "gguf"
        if any(item.get("id") == model_id for item in self.config.local_models):
            raise ValueError(f"A local model with id '{model_id}' already exists")

        entry = {
            "id": model_id,
            "name": name or model_id,
            "engine": engine,
            "format": fmt,
            "path": str(resolved),
            "modalities": modalities or ["text"],
            "size_gb": round(directory_size(resolved), 2),
            "license": "local",
            "notes": "Locally registered model",
        }
        self.config.local_models.append(entry)
        save_config(self.config)
        return find_spec(self.config, model_id)

    def remove_local(self, model_id: str) -> None:
        before = len(self.config.local_models)
        self.config.local_models = [
            item for item in self.config.local_models if item.get("id") != model_id
        ]
        if len(self.config.local_models) == before:
            raise KeyError(f"No local model with id '{model_id}'")
        save_config(self.config)

    def delete_files(self, spec: ModelSpec) -> None:
        if spec.is_local:
            raise ValueError("Refusing to delete files for a locally registered model")
        path = self.path_for(spec)
        if not path.exists():
            return
        models_root = self.config.models_path.resolve()
        target = path.resolve()
        if models_root not in target.parents and target != models_root:
            raise ValueError(f"Refusing to delete a path outside the models directory: {target}")
        shutil.rmtree(target)

    def image_path(self, spec: ImageModelSpec) -> Path:
        return self.config.models_path / spec.dir_name

    @staticmethod
    def path_is_image_model(path: Path) -> bool:
        required = {
            "model_index.json": 1,
            "unet/openvino_model.xml": 1,
            "unet/openvino_model.bin": 500_000_000,
            "text_encoder/openvino_model.bin": 50_000_000,
            "vae_decoder/openvino_model.bin": 20_000_000,
        }
        for relative, minimum in required.items():
            candidate = path / relative
            if not candidate.exists() or candidate.stat().st_size < minimum:
                return False
        return True

    def image_installed(self, spec: ImageModelSpec) -> bool:
        return self.path_is_image_model(self.image_path(spec))

    def download_image(self, spec: ImageModelSpec, force: bool = False) -> Path:
        if self.image_installed(spec) and not force:
            print(f"{spec.id} is already installed at {self.image_path(spec)}")
            return self.image_path(spec)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is not installed") from exc
        target = self.image_path(spec)
        target.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {spec.name} ({spec.size_gb:.2f} GB) from {spec.repo} ...")
        try:
            snapshot_download(repo_id=spec.repo, local_dir=str(target))
        except Exception as exc:
            raise RuntimeError(
                f"Image model download failed for {spec.repo}: {exc}. "
                "Run setup again to resume."
            ) from exc
        if not self.image_installed(spec):
            raise RuntimeError(
                f"Download finished but no OpenVINO image model was found in {target}."
            )
        print(f"Installed {spec.id} at {target}")
        return target
