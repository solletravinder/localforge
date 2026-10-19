from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET_ROOT = ROOT / "portable"
TARGET = TARGET_ROOT / "LocalForge"
VENV = ROOT / ".venv"
SITE = VENV / "Lib" / "site-packages"
PYPROJECT = ROOT / "pyproject.toml"
EXE = ROOT / "LocalForge.exe"
SANITY_REPO = "DevQuasar/huihui-ai.Qwen3-0.6B-abliterated-GGUF"
SANITY_DIR = "qwen3-0.6b-abliterated-gguf"


def read_version() -> str:
    for line in PYPROJECT.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("version ="):
            return line.split("=", 1)[1].strip().strip('"')
    return "0.0.0"


def read_base_python() -> Path:
    cfg = VENV / "pyvenv.cfg"
    for line in cfg.read_text(encoding="utf-8").splitlines():
        if line.lower().startswith("home ="):
            return Path(line.split("=", 1)[1].strip())
    raise RuntimeError(f"Cannot find base Python from {cfg}")


def find_llama_package() -> Path:
    configured = os.environ.get("LOCALFORGE_LLAMA_DIR")
    if configured:
        matches = sorted(Path(configured).expanduser().rglob("llama-server.exe"))
        if matches:
            return matches[0].parent
        raise RuntimeError(f"llama-server.exe was not found under {configured}")
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    packages = local / "Microsoft" / "WinGet" / "Packages"
    matches = sorted(packages.glob("ggml.llamacpp*/llama-server.exe"))
    if not matches:
        raise RuntimeError(
            "llama-server.exe was not found. Set LOCALFORGE_LLAMA_DIR or install the WinGet package."
        )
    return matches[0].parent


def ignore_site(directory: str, names: list[str]) -> set[str]:
    skipped = set()
    for name in names:
        lower = name.lower()
        if lower in {"local_forge", "local_forge-0.1.0.dist-info"}:
            skipped.add(name)
        elif lower.startswith("__editable__.local_forge"):
            skipped.add(name)
        elif lower in {"pyinstaller", "pyinstaller_hooks_contrib", "_pytest", "pytest"}:
            skipped.add(name)
        elif lower.endswith(".dist-info") and (
            lower.startswith("pyinstaller") or lower.startswith("pytest")
        ):
            skipped.add(name)
    return skipped


def ignore_app(directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name == "__pycache__" or name.endswith(".pyc")}


def main() -> None:
    version = read_version()
    if not EXE.is_file():
        raise SystemExit("Build LocalForge.exe first with build-exe.bat")
    if not SITE.is_dir():
        raise SystemExit(f"Missing virtual environment packages: {SITE}")

    if TARGET.exists():
        print(f"Removing old portable tree: {TARGET}")
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True)
    runtime = TARGET / "runtime"
    python_dir = runtime / "python"
    print("Copying Python runtime...")
    shutil.copytree(read_base_python(), python_dir, ignore=shutil.ignore_patterns("site-packages"))

    target_site = python_dir / "Lib" / "site-packages"
    if target_site.exists():
        shutil.rmtree(target_site)
    print("Copying installed dependencies...")
    shutil.copytree(SITE, target_site, ignore=ignore_site)

    app_target = runtime / "app" / "local_forge"
    print("Copying LocalForge package...")
    shutil.copytree(ROOT / "src" / "local_forge", app_target, ignore=ignore_app)

    llama_target = runtime / "llama"
    llama_target.mkdir(parents=True)
    llama_source = find_llama_package()
    print(f"Copying llama.cpp runtime from {llama_source}...")
    for item in llama_source.iterdir():
        if not item.is_file():
            continue
        lower = item.name.lower()
        if lower.endswith(".dll") or lower in {"llama-server.exe", "llama-server-impl.dll"}:
            shutil.copy2(item, llama_target / item.name)

    shutil.copy2(EXE, TARGET / "LocalForge.exe")
    (TARGET / "portable.flag").write_text("LocalForge portable mode\n", encoding="utf-8")
    models_dir = TARGET / "models"
    models_dir.mkdir()
    print("Downloading bundled Qwen 0.6B sanity model...")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=SANITY_REPO,
        local_dir=str(models_dir / SANITY_DIR),
        allow_patterns=["*Q4_K_M.gguf", "*q4_k_m.gguf"],
    )
    data = TARGET / "data"
    data.mkdir()
    config = {
        "openvino_device": "CPU",
        "llama_server": "runtime/llama/llama-server.exe",
        "llama_ngl": 10,
        "llama_ctx": 131072,
        "llama_batch": 2048,
        "llama_ubatch": 1024,
        "llama_threads": 0,
        "llama_parallel": 1,
        "llama_flash_attn": "on",
        "llama_cache_k": "q8_0",
        "llama_cache_v": "q4_0",
        "api_host": "127.0.0.1",
        "api_port": 8090,
        "image_model": "",
        "system_prompt": "You are a helpful local assistant. Follow the user's instructions.",
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "warmup": True,
        "disable_thinking": True,
        "first_run_notice": False,
        "local_models": [],
    }
    (data / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (TARGET / "README-PORTABLE.txt").write_text(
        "LocalForge Portable "
        + version
        + "\n\n"
        "Double-click LocalForge.exe. Python and the model server are bundled; no terminal is\n"
        "needed.\n\n"
        "The package includes an abliterated Qwen3 0.6B sanity model at 8K context. On first\n"
        "run it is selected automatically: click Start Server and send a short message to\n"
        "verify the installation. Then open Models and download Qwen3.5 4B for the full 131k\n"
        "profile. Models are stored in the models folder next to LocalForge.exe.\n\n"
        "Jan is optional. Point it at http://127.0.0.1:8090/v1 with model\n"
        "qwen3.5-4b-abliterated and context 131000.\n\n"
        "Folders:\n"
        "  runtime/python  bundled Python and dependencies\n"
        "  runtime/llama   bundled llama-server.exe\n"
        "  runtime/app     LocalForge package\n"
        "  data            config and logs\n"
        "  models          downloaded models\n",
        encoding="utf-8",
    )
    print(f"Built portable app: {TARGET}")
    print("Zip it or compile installer.iss with Inno Setup.")


if __name__ == "__main__":
    main()
