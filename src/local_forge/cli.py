from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .catalog import DEFAULT_MODEL_ID, IMAGE_CATALOG, ModelSpec, find_spec
from .config import (
    AppConfig,
    DISCLAIMER,
    config_path,
    load_config,
    mark_notice_seen,
    save_config,
    set_config_value,
)
from .engines import ChatOptions, EngineError, EngineManager, EngineUnavailable
from .hardware import detect_hardware, format_gb
from .manager import ModelManager
from .schemas import load_image_file


def _first_run(config: AppConfig) -> None:
    if config.first_run_notice:
        print("=" * 78)
        print(DISCLAIMER)
        print("=" * 78)
        mark_notice_seen(config)


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    string_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in string_rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in string_rows:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


def _pick_model(config: AppConfig, manager: ModelManager, model_id: str | None) -> ModelSpec:
    if model_id:
        return manager.get(model_id)
    installed = [spec for spec in manager.specs() if manager.installed(spec)]
    if not installed:
        return manager.get(DEFAULT_MODEL_ID)
    preferred = next((spec for spec in installed if spec.id == DEFAULT_MODEL_ID), None)
    return preferred or installed[0]


def _save_images(images, out: str | None, seed: int) -> list[Path]:
    if out:
        target = Path(out).expanduser()
        if target.exists() and target.is_dir():
            target = target / f"forge-{int(time.time())}.png"
    else:
        target = Path("outputs") / f"forge-{int(time.time())}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if len(images) == 1:
        image_path = target if target.suffix else target.with_suffix(".png")
        images[0].save(image_path)
        paths.append(image_path)
    else:
        base = target.with_suffix("") if target.suffix else target
        for index, image in enumerate(images, start=1):
            image_path = base.with_name(f"{base.name}-{index}-seed{seed}.png")
            image.save(image_path)
            paths.append(image_path)
    return paths


def cmd_setup(args, config: AppConfig) -> int:
    _first_run(config)
    if args.text_only and args.image_only:
        raise ValueError("Use only one of --text-only or --image-only")
    manager = ModelManager(config)
    text_missing = (
        []
        if args.image_only
        else [
            spec
            for spec in manager.specs()
            if not spec.is_local and not manager.installed(spec)
        ]
    )
    image_missing = (
        [] if args.text_only else [spec for spec in IMAGE_CATALOG if not manager.image_installed(spec)]
    )
    image_spec = IMAGE_CATALOG[0]
    image_path = manager.image_path(image_spec)

    if not text_missing and not image_missing:
        if not args.text_only:
            config.image_model = str(image_path)
            save_config(config)
        print("Everything is already installed.")
        print(f"  chat:  run.bat chat")
        print(f"  image: run.bat image \"your prompt\"")
        return 0

    total = sum(spec.size_gb for spec in text_missing) + sum(
        spec.size_gb for spec in image_missing
    )
    print("LocalForge setup will install:")
    for spec in text_missing:
        print(f"  text  {spec.id:<28} {spec.size_gb:>5.2f} GB")
    for spec in image_missing:
        print(f"  image {spec.id:<28} {spec.size_gb:>5.2f} GB")
    print(f"Total: {total:.2f} GB")
    print(
        "The download resumes if interrupted. On a slower connection this can take a while; "
        "run this command again to continue."
    )
    if not args.yes:
        answer = input("Continue? [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            print("Cancelled.")
            return 0

    failures: list[str] = []
    for spec in text_missing:
        try:
            manager.download(spec)
        except Exception as exc:
            failures.append(f"{spec.id}: {exc}")
            print(f"FAILED: {spec.id}: {exc}", file=sys.stderr)
    if not args.text_only:
        try:
            target = manager.download_image(image_spec)
            config.image_model = str(target)
            save_config(config)
        except Exception as exc:
            failures.append(f"{image_spec.id}: {exc}")
            print(f"FAILED: {image_spec.id}: {exc}", file=sys.stderr)

    print()
    if failures:
        print("Some downloads did not finish. Run 'run.bat setup' again to resume.", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("Setup complete.")
    print("  Chat:  run.bat chat")
    print("  Image: run.bat image \"your prompt\"")
    print("  API:   run.bat serve")
    return 0


def cmd_doctor(args, config: AppConfig) -> int:
    info = detect_hardware(config)
    print(f"LocalForge {__version__} doctor")
    print(f"config:      {config_path()}")
    print(f"models dir:  {config.models_path}")
    print(f"OS:          {info['os']}")
    print(f"CPU:         {info['cpu']} ({info['cpu_cores']} threads)")
    print(f"RAM:         {info['ram_gb']} GB")
    print(f"GPU:         {', '.join(info['gpus']) or 'not detected'}")
    print(
        "OpenVINO:    "
        + (", ".join(info["openvino_devices"]) or "no devices (check the install)")
    )
    print(f"llama.cpp:   {info['llama_server'] or 'not found'}")
    print(f"llama ver:   {info['llama_server_version']}")
    try:
        import openvino as ov

        print(f"OV runtime:  {getattr(ov, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"OV runtime:  import failed ({exc})")
    try:
        import openvino_genai as ovg

        print(f"OV GenAI:    {getattr(ovg, '__version__', 'installed')}")
    except Exception as exc:
        print(f"OV GenAI:    import failed ({exc})")
    print(f"device cfg:  {config.openvino_device}")
    image_path = config.image_model_path
    image_status = "not configured"
    if image_path:
        installed_image = ModelManager.path_is_image_model(image_path)
        image_status = f"{image_path} ({'installed' if installed_image else 'incomplete or missing'})"
    print("image model: " + image_status)
    print()
    manager = ModelManager(config)
    rows = []
    for spec in manager.specs():
        status = manager.status(spec)
        rows.append(
            [
                spec.id,
                spec.engine,
                "yes" if status["installed"] else "no",
                format_gb(status["size_gb"] if status["installed"] else spec.size_gb),
                ",".join(spec.modalities),
            ]
        )
    _print_table(["MODEL", "ENGINE", "INSTALLED", "SIZE", "MODALITIES"], rows)
    missing = [spec for spec in manager.specs() if not manager.installed(spec)]
    if missing:
        print("\nInstall missing models with:")
        for spec in missing:
            print(f"  lforge models download {spec.id}")
    return 0


def cmd_models_list(args, config: AppConfig) -> int:
    manager = ModelManager(config)
    if getattr(args, "json", False):
        print(json.dumps([manager.status(spec) for spec in manager.specs()], indent=2))
        return 0
    rows = []
    for spec in manager.specs():
        status = manager.status(spec)
        rows.append(
            [
                spec.id,
                spec.engine,
                "yes" if status["installed"] else "no",
                format_gb(status["size_gb"] if status["installed"] else spec.size_gb),
                ",".join(spec.modalities),
                spec.license,
            ]
        )
    _print_table(["MODEL", "ENGINE", "INSTALLED", "SIZE", "MODALITIES", "LICENSE"], rows)
    print(f"\nmodels dir: {config.models_path}")
    return 0


def cmd_models_download(args, config: AppConfig) -> int:
    manager = ModelManager(config)
    spec = manager.get(args.model_id)
    manager.download(spec, force=args.force)
    status = manager.status(spec)
    print(json.dumps(status, indent=2))
    return 0


def cmd_models_remove(args, config: AppConfig) -> int:
    manager = ModelManager(config)
    spec = find_spec(config, args.model_id)
    if spec.is_local:
        manager.remove_local(spec.id)
        print(f"Unregistered local model {spec.id}; files were not deleted.")
        return 0
    if not manager.installed(spec):
        print(f"{spec.id} is not installed.")
        return 0
    path = manager.path_for(spec)
    size = manager.size_gb(spec)
    if not args.yes:
        answer = input(f"Delete {spec.id} ({size:.2f} GB) at {path}? [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            print("Cancelled.")
            return 0
    manager.delete_files(spec)
    print(f"Deleted {path}")
    return 0


def cmd_models_add_local(args, config: AppConfig) -> int:
    manager = ModelManager(config)
    modalities = [item.strip() for item in args.modalities.split(",") if item.strip()]
    spec = manager.add_local(
        args.model_id,
        args.path,
        args.engine,
        name=args.name,
        modalities=modalities,
    )
    print(f"Registered {spec.id} at {spec.local_path}")
    return 0


def cmd_models_inspect(args, config: AppConfig) -> int:
    from .gguf_info import inspect_gguf

    try:
        info = inspect_gguf(args.path)
    except Exception as exc:
        print(f"Inspect failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(info, indent=2))
    return 0


def cmd_models_download_url(args, config: AppConfig) -> int:
    from .url_download import download_url, engine_for_file, filename_from_url

    model_id = args.id.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model_id):
        print(
            "Model ID may contain letters, numbers, dots, underscores, and hyphens.",
            file=sys.stderr,
        )
        return 1
    manager = ModelManager(config)
    if any(spec.id == model_id for spec in manager.specs()):
        print(f"A model with id '{model_id}' already exists.", file=sys.stderr)
        return 1
    modalities = [item.strip() for item in args.modalities.split(",") if item.strip()]
    if not modalities:
        modalities = ["text"]
    try:
        resolved_filename = filename_from_url(args.url, args.filename)
        engine = args.engine
        if engine == "auto":
            engine = engine_for_file(resolved_filename)
        target_dir = config.models_path / model_id
        path = download_url(args.url, target_dir, args.filename)
        spec = manager.add_local(
            model_id,
            str(path),
            engine,
            name=args.name or model_id,
            modalities=modalities,
        )
    except Exception as exc:
        print(f"URL download failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manager.status(spec), indent=2))
    return 0


def cmd_models_status(args, config: AppConfig) -> int:
    manager = ModelManager(config)
    spec = find_spec(config, args.model_id)
    print(json.dumps(manager.status(spec), indent=2))
    return 0


def cmd_config(args, config: AppConfig) -> int:
    if args.config_command == "show":
        print(json.dumps(asdict(config), indent=2))
    elif args.config_command == "path":
        print(config_path())
    elif args.config_command == "set":
        set_config_value(config, args.key, args.value)
        print(f"{args.key} = {getattr(config, args.key)}")
    elif args.config_command == "init":
        if args.models_dir:
            config.models_dir = args.models_dir
        config.ensure_dirs()
        save_config(config)
        print(f"Wrote {config_path()}")
    else:
        raise ValueError("Unknown config command")
    return 0


def cmd_chat(args, config: AppConfig) -> int:
    manager = ModelManager(config)
    spec = _pick_model(config, manager, args.model)
    if not manager.installed(spec):
        print(
            f"{spec.id} is not installed.\nRun: lforge models download {spec.id}",
            file=sys.stderr,
        )
        return 1
    _first_run(config)
    print(f"Model:  {spec.id} ({spec.engine}, {','.join(spec.modalities)})")
    print("Type /help for commands. Ctrl+C or /quit to exit.")
    engine_manager = EngineManager(config, manager)
    system_prompt = args.system if args.system is not None else config.system_prompt
    history: list[dict] = []
    images: list[bytes] = []
    for image_path in args.image or []:
        images.append(load_image_file(image_path))
        print(f"Attached image: {image_path}")
    try:
        engine = engine_manager.get_chat_engine(spec.id)
        while True:
            try:
                raw = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not raw:
                continue
            if raw.startswith("/"):
                command, _, rest = raw.partition(" ")
                command = command.lower()
                rest = rest.strip()
                if command in {"/quit", "/exit"}:
                    break
                if command == "/help":
                    print(
                        "/model <id>   switch model\n"
                        "/image <path> attach an image to the next message\n"
                        "/system <s>   set the system prompt\n"
                        "/reset        clear conversation and images\n"
                        "/quit         exit"
                    )
                    continue
                if command == "/reset":
                    history.clear()
                    images.clear()
                    print("Conversation cleared.")
                    continue
                if command == "/system":
                    system_prompt = rest
                    print("System prompt updated.")
                    continue
                if command == "/image":
                    if not rest:
                        print("Usage: /image <path>")
                        continue
                    path = Path(rest.strip('"').strip("'")).expanduser()
                    images.append(load_image_file(str(path)))
                    print(f"Attached {path} ({len(images)} image(s) pending)")
                    continue
                if command == "/model":
                    if not rest:
                        print("Usage: /model <id>")
                        continue
                    try:
                        new_spec = manager.get(rest.strip())
                        if not manager.installed(new_spec):
                            print(f"{new_spec.id} is not installed.")
                            continue
                        engine = engine_manager.get_chat_engine(new_spec.id)
                        spec = new_spec
                        history.clear()
                        images.clear()
                        print(f"Switched to {spec.id}. Conversation cleared.")
                    except (KeyError, EngineError, EngineUnavailable) as exc:
                        print(f"error: {exc}")
                    continue
                print(f"Unknown command: {command}. Type /help.")
                continue

            history.append({"role": "user", "content": raw})
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(history)
            options = ChatOptions(
                max_tokens=args.max_tokens or config.max_new_tokens,
                temperature=args.temperature if args.temperature is not None else config.temperature,
                top_p=args.top_p if args.top_p is not None else config.top_p,
                chat_template_kwargs={"enable_thinking": False} if args.no_thinking else None,
            )
            try:
                if args.no_stream:
                    text = engine.chat(messages, images, options)
                    print(f"\nassistant> {text}")
                else:
                    print("\nassistant> ", end="", flush=True)
                    parts: list[str] = []
                    for chunk in engine.stream_chat(messages, images, options):
                        parts.append(chunk)
                        print(chunk, end="", flush=True)
                    print()
                    text = "".join(parts)
                if not text.strip():
                    text = "(empty response)"
                history.append({"role": "assistant", "content": text})
                images.clear()
            except (EngineError, EngineUnavailable, ValueError) as exc:
                print(f"\nerror: {exc}", file=sys.stderr)
                history.pop()
    finally:
        engine_manager.close()
    return 0


def cmd_serve(args, config: AppConfig) -> int:
    _first_run(config)
    manager = ModelManager(config)
    engine_manager = EngineManager(config, manager)
    host = args.host or config.api_host
    port = args.port or config.api_port
    if host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            "WARNING: binding to a non-local address exposes an unauthenticated API "
            "with no content filter.",
            file=sys.stderr,
        )
    if args.preload:
        spec = _pick_model(config, manager, args.model)
        if not manager.installed(spec):
            print(f"{spec.id} is not installed. Run: lforge models download {spec.id}", file=sys.stderr)
            return 1
        engine_manager.get_chat_engine(spec.id)

    from .api import create_app

    app = create_app(config, manager, engine_manager)
    print(f"LocalForge API: http://{host}:{port}")
    print(f"OpenAI base URL: http://{host}:{port}/v1")
    print(f"Docs: http://{host}:{port}/docs")
    try:
        import uvicorn

        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        engine_manager.close()
    return 0


def cmd_image(args, config: AppConfig) -> int:
    _first_run(config)
    model_path = args.model_path or config.image_model
    if not model_path:
        print(
            "No image model configured.\n"
            "Export an uncensored SD1.5/SDXL checkpoint to OpenVINO IR first:\n"
            "  lforge convert-image --model <hf-repo-or-local-diffusers-dir> --out models/sd15-ov --set-config\n"
            "Then run:\n"
            "  lforge image \"your prompt\"",
            file=sys.stderr,
        )
        return 1
    manager = ModelManager(config)
    engine_manager = EngineManager(config, manager)
    try:
        engine = engine_manager.get_image_engine(model_path)
        result = engine.generate(
            prompt=args.prompt,
            negative_prompt=args.negative,
            width=args.width or config.image_width,
            height=args.height or config.image_height,
            steps=args.steps or config.image_steps,
            guidance_scale=args.guidance if args.guidance is not None else config.image_guidance,
            seed=args.seed,
            num_images=args.n,
        )
        paths = _save_images(result.images, args.out, result.seed)
        print(f"seed={result.seed} size={result.width}x{result.height}")
        for path in paths:
            print(f"saved: {path}")
    finally:
        engine_manager.close()
    return 0


def cmd_convert_image(args, config: AppConfig) -> int:
    optimum = shutil.which("optimum-cli")
    if not optimum:
        print(
            "optimum-cli was not found. Install it with:\n"
            "  pip install \"optimum-intel[openvino]\"\n"
            "Then rerun this command.",
            file=sys.stderr,
        )
        return 1
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        optimum,
        "export",
        "openvino",
        "--model",
        args.model,
        "--task",
        "text-to-image",
        "--weight-format",
        args.weight_format,
    ]
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    command.append(str(out))
    print("Running:", " ".join(command))
    result = subprocess.run(command)
    if result.returncode != 0:
        print("Conversion failed.", file=sys.stderr)
        return result.returncode
    print(f"Exported image model to {out}")
    if args.set_config:
        config.image_model = str(out)
        save_config(config)
        print(f"image_model set to {out}")
    print(f'Try: lforge image "a moody portrait" --model-path "{out}"')
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lforge",
        description="LocalForge: local uncensored text/vision/image generation on Windows.",
    )
    parser.add_argument("--models-dir", help="Override the models directory for this run")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup", help="Download all models and configure image defaults")
    setup.add_argument("--text-only", action="store_true")
    setup.add_argument("--image-only", action="store_true")
    setup.add_argument("--yes", action="store_true")
    setup.set_defaults(func=cmd_setup)

    doctor = sub.add_parser("doctor", help="Show hardware, engines, and model status")
    doctor.set_defaults(func=cmd_doctor)

    models = sub.add_parser("models", help="Manage models")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    models_list = models_sub.add_parser("list", help="List catalog models")
    models_list.add_argument("--json", action="store_true", help="Print machine-readable model status")
    models_list.set_defaults(func=cmd_models_list)
    models_download = models_sub.add_parser("download", help="Download a model")
    models_download.add_argument("model_id")
    models_download.add_argument("--force", action="store_true")
    models_download.set_defaults(func=cmd_models_download)
    models_remove = models_sub.add_parser("remove", help="Delete a model or unregister a local one")
    models_remove.add_argument("model_id")
    models_remove.add_argument("--yes", action="store_true")
    models_remove.set_defaults(func=cmd_models_remove)
    models_add = models_sub.add_parser("add-local", help="Register a local GGUF, OpenVINO, or LiteRT model")
    models_add.add_argument("model_id")
    models_add.add_argument("path")
    models_add.add_argument("--engine", choices=["llama.cpp", "openvino", "litert"], required=True)
    models_add.add_argument("--name")
    models_add.add_argument("--modalities", default="text", help="Comma-separated: text,image")
    models_add.set_defaults(func=cmd_models_add_local)
    models_status = models_sub.add_parser("status", help="Show one model's status")
    models_status.add_argument("model_id")
    models_status.set_defaults(func=cmd_models_status)
    models_inspect = models_sub.add_parser("inspect", help="Inspect GGUF metadata and recommend llama.cpp settings")
    models_inspect.add_argument("path")
    models_inspect.set_defaults(func=cmd_models_inspect)
    models_url = models_sub.add_parser("download-url", help="Download a GGUF or LiteRT model from a direct URL")
    models_url.add_argument("url")
    models_url.add_argument("--id", required=True)
    models_url.add_argument("--name")
    models_url.add_argument("--engine", choices=["auto", "llama.cpp", "litert"], default="auto")
    models_url.add_argument("--modalities", default="text", help="Comma-separated: text,image")
    models_url.add_argument("--filename", help="Optional filename if the URL has no clean name")
    models_url.set_defaults(func=cmd_models_download_url)

    config = sub.add_parser("config", help="Show or edit configuration")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    config_show = config_sub.add_parser("show")
    config_show.set_defaults(func=cmd_config)
    config_path_cmd = config_sub.add_parser("path")
    config_path_cmd.set_defaults(func=cmd_config)
    config_set = config_sub.add_parser("set")
    config_set.add_argument("key")
    config_set.add_argument("value")
    config_set.set_defaults(func=cmd_config)
    config_init = config_sub.add_parser("init")
    config_init.add_argument("--models-dir")
    config_init.set_defaults(func=cmd_config)

    chat = sub.add_parser("chat", help="Interactive streaming chat")
    chat.add_argument("--model")
    chat.add_argument("--system")
    chat.add_argument("--image", action="append", help="Attach an image (repeatable)")
    chat.add_argument("--max-tokens", type=int)
    chat.add_argument("--temperature", type=float)
    chat.add_argument("--top-p", type=float)
    chat.add_argument("--no-stream", action="store_true")
    chat.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable reasoning traces for models whose chat template supports enable_thinking "
        "(currently llama.cpp models such as Qwen3.5).",
    )
    chat.set_defaults(func=cmd_chat)

    serve = sub.add_parser("serve", help="Run the OpenAI-compatible local API")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--preload", action="store_true")
    serve.add_argument("--model")
    serve.set_defaults(func=cmd_serve)

    image = sub.add_parser("image", help="Generate an image")
    image.add_argument("prompt")
    image.add_argument("--model-path")
    image.add_argument("--out")
    image.add_argument("--width", type=int)
    image.add_argument("--height", type=int)
    image.add_argument("--steps", type=int)
    image.add_argument("--guidance", type=float)
    image.add_argument("--negative")
    image.add_argument("--seed", type=int)
    image.add_argument("--n", type=int, default=1)
    image.set_defaults(func=cmd_image)

    convert = sub.add_parser("convert-image", help="Export a diffusers SD checkpoint to OpenVINO IR")
    convert.add_argument("--model", required=True, help="Hugging Face repo id or local diffusers directory")
    convert.add_argument("--out", required=True)
    convert.add_argument("--weight-format", default="int8", choices=["int8", "int4", "fp16"])
    convert.add_argument("--trust-remote-code", action="store_true")
    convert.add_argument("--set-config", action="store_true")
    convert.set_defaults(func=cmd_convert_image)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config()
        if getattr(args, "models_dir", None):
            config.models_dir = args.models_dir
            config.ensure_dirs()
        return int(args.func(args, config) or 0)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError, KeyError, EngineError, EngineUnavailable) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
