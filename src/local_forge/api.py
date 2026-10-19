from __future__ import annotations

import base64
import io
import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from . import __version__
from .catalog import find_spec
from .config import AppConfig, load_config
from .engines import ChatOptions, EngineError, EngineManager, EngineUnavailable
from .manager import ModelManager
from .schemas import (
    ChatCompletionRequest,
    ImageGenerationRequest,
    prepare_messages,
)
from .tools import prepare_openai_tool_messages, prepare_tool_messages


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def _resolve_spec(config: AppConfig, manager: ModelManager, model_id: str | None):
    if model_id:
        spec = find_spec(config, model_id)
    else:
        specs = manager.specs()
        spec = next((item for item in specs if manager.installed(item)), specs[0])
    if not manager.installed(spec):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Model '{spec.id}' is not installed. Run: "
                f"lforge models download {spec.id}"
            ),
        )
    return spec


def _engine_error(exc: Exception) -> HTTPException:
    if isinstance(exc, EngineUnavailable):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, EngineError):
        return HTTPException(status_code=502, detail=str(exc))
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc).strip("'"))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


def create_app(
    config: AppConfig | None = None,
    model_manager: ModelManager | None = None,
    engine_manager: EngineManager | None = None,
) -> FastAPI:
    config = config or load_config()
    model_manager = model_manager or ModelManager(config)
    engine_manager = engine_manager or EngineManager(config, model_manager)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        engine_manager.close()

    app = FastAPI(
        title="LocalForge",
        version=__version__,
        description="Local uncensored multimodal inference with OpenVINO GPU and llama.cpp.",
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.models = model_manager
    app.state.engines = engine_manager
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root() -> dict:
        return {
            "name": "LocalForge",
            "version": __version__,
            "models_dir": str(config.models_path),
            "endpoints": [
                "/v1/models",
                "/v1/chat/completions",
                "/v1/images/generations",
                "/health",
                "/docs",
            ],
            "disclaimer": (
                "Uncensored models have no content filter. You are responsible for generated "
                "content and compliance with applicable law."
            ),
        }

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "app": "localforge",
            "version": __version__,
            "models_dir": str(config.models_path),
            "engines": engine_manager.status(),
        }

    @app.get("/v1/models")
    def list_models(installed_only: bool = Query(False)) -> dict:
        data = []
        for spec in model_manager.specs():
            status = model_manager.status(spec)
            if installed_only and not status["installed"]:
                continue
            data.append(
                {
                    "id": spec.id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                    "engine": spec.engine,
                    "installed": status["installed"],
                    "modalities": status["modalities"],
                    "license": status["license"],
                    "size_gb": status["size_gb"],
                    "path": status["path"],
                    "notes": spec.notes,
                }
            )
        return {"object": "list", "data": data}

    @app.get("/v1/models/{model_id}")
    def get_model(model_id: str) -> dict:
        try:
            spec = find_spec(config, model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc).strip("'")) from exc
        status = model_manager.status(spec)
        return {
            "id": spec.id,
            "object": "model",
            "owned_by": "local",
            **status,
        }

    def _chat_options(request: ChatCompletionRequest) -> ChatOptions:
        max_tokens = request.max_completion_tokens or request.max_tokens or config.max_new_tokens
        template_kwargs = dict(request.chat_template_kwargs or {})
        if request.enable_thinking is not None:
            template_kwargs["enable_thinking"] = request.enable_thinking
        return ChatOptions(
            max_tokens=int(max_tokens),
            temperature=request.temperature if request.temperature is not None else config.temperature,
            top_p=request.top_p if request.top_p is not None else config.top_p,
            seed=request.seed,
            chat_template_kwargs=template_kwargs or None,
            tools=request.tools,
            tool_choice=request.tool_choice,
        )

    def _has_tool_context(request: ChatCompletionRequest) -> bool:
        if request.tools:
            return True
        for message in request.messages:
            if (message.role or "").strip().lower() == "tool":
                return True
            if (message.model_extra or {}).get("tool_calls"):
                return True
            content = message.content
            if isinstance(content, list) and any(
                isinstance(part, dict) and part.get("type") == "tool_call"
                for part in content
            ):
                return True
        return False

    def _tool_message(request: ChatCompletionRequest, result) -> dict:
        message: dict = {"role": "assistant"}
        if result.content:
            message["content"] = result.content
        else:
            message["content"] = None
        if result.tool_calls:
            message["tool_calls"] = result.tool_calls
        return message

    def _tool_sse(result) -> str:
        completion_id = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())

        def chunk(delta: dict, finish_reason: str | None = None) -> str:
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": result.model_id,
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish_reason}
                ],
            }
            return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

        yield chunk({"role": "assistant", "content": ""})
        if result.tool_calls:
            tool_deltas = []
            for index, call in enumerate(result.tool_calls):
                tool_deltas.append(
                    {
                        "index": index,
                        "id": call["id"],
                        "type": "function",
                        "function": call["function"],
                    }
                )
            yield chunk({"tool_calls": tool_deltas}, "tool_calls")
        else:
            text = result.content or ""
            for start in range(0, len(text), 48):
                yield chunk({"content": text[start : start + 48]})
            yield chunk({}, "stop")
        yield "data: [DONE]\n\n"

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        tool_context = _has_tool_context(request)
        try:
            spec = _resolve_spec(config, model_manager, request.model)
            options = _chat_options(request)
            prepared = None
            if not (tool_context and spec.engine == "llama.cpp"):
                prepared = prepare_messages(request.messages)
            engine = engine_manager.get_chat_engine(spec.id)
        except (HTTPException, EngineError, EngineUnavailable, ValueError, KeyError) as exc:
            if isinstance(exc, HTTPException):
                raise
            raise _engine_error(exc) from exc

        if tool_context:
            if spec.engine in {"openvino", "litert"}:
                if spec.engine == "litert" and not getattr(engine, "supports_tools", False):
                    raise HTTPException(
                        status_code=501,
                        detail=(
                            "This LiteRT bundle did not pass native tool-call validation. "
                            "Use a compatible bundle; raw tool text is not parsed."
                        ),
                    )
                try:
                    if spec.engine == "openvino":
                        tool_messages = prepare_tool_messages(request.messages)
                    else:
                        tool_messages = prepare_openai_tool_messages(request.messages)
                    result = engine.chat_tools(
                        tool_messages,
                        prepared.images if prepared else [],
                        options,
                        request.tools,
                    )
                except (EngineError, EngineUnavailable, ValueError) as exc:
                    raise _engine_error(exc) from exc
                result.model_id = spec.id
                if request.stream:
                    return StreamingResponse(
                        _tool_sse(result),
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                    )
                completion_id = "chatcmpl-" + uuid.uuid4().hex
                created = int(time.time())
                text = result.content or ""
                return {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": spec.id,
                    "choices": [
                        {
                            "index": 0,
                            "message": _tool_message(request, result),
                            "finish_reason": "tool_calls" if result.tool_calls else "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": _estimate_tokens(text),
                        "total_tokens": _estimate_tokens(text),
                        "estimated": True,
                    },
                }
            if spec.engine == "llama.cpp":
                raw_messages = prepare_openai_tool_messages(request.messages)
                if request.stream:
                    return StreamingResponse(
                        engine.stream_raw(raw_messages, options),
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                    )
                try:
                    return engine.chat_raw(raw_messages, options)
                except (EngineError, EngineUnavailable, ValueError) as exc:
                    raise _engine_error(exc) from exc
            raise HTTPException(
                status_code=501,
                detail="Tool calling is not supported for this engine.",
            )

        if request.stream:
            return StreamingResponse(
                _stream_completion(engine, spec.id, prepared, options),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            text = engine.chat(prepared.messages, prepared.images, options)
        except (EngineError, EngineUnavailable, ValueError) as exc:
            raise _engine_error(exc) from exc
        completion_id = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        prompt_tokens = _estimate_tokens(
            " ".join(message.get("content", "") for message in prepared.messages)
        )
        completion_tokens = _estimate_tokens(text)
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": spec.id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "estimated": True,
            },
        }

    def _stream_completion(engine, model_id: str, prepared, options: ChatOptions):
        completion_id = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())

        def chunk(delta: dict, finish_reason: str | None = None) -> str:
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish_reason}
                ],
            }
            return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

        yield chunk({"role": "assistant", "content": ""})
        try:
            for token in engine.stream_chat(prepared.messages, prepared.images, options):
                if token:
                    yield chunk({"content": token})
        except (EngineError, EngineUnavailable, ValueError) as exc:
            error = {"error": {"message": str(exc), "type": "engine_error"}}
            yield "data: " + json.dumps(error, ensure_ascii=False) + "\n\n"
        yield chunk({}, "stop")
        yield "data: [DONE]\n\n"

    @app.post("/v1/images/generations")
    def generate_images(request: ImageGenerationRequest):
        if request.response_format != "b64_json":
            raise HTTPException(
                status_code=400,
                detail="Only response_format='b64_json' is supported by the local API.",
            )
        try:
            width_text, height_text = request.size.lower().split("x", 1)
            width, height = int(width_text), int(height_text)
        except (ValueError, AttributeError) as exc:
            raise HTTPException(status_code=400, detail="size must look like 512x512") from exc
        requested_model = request.model if request.model not in (None, "", "default") else None
        try:
            engine = engine_manager.get_image_engine(requested_model)
            result = engine.generate(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=width,
                height=height,
                steps=request.steps or config.image_steps,
                guidance_scale=(
                    request.guidance_scale
                    if request.guidance_scale is not None
                    else config.image_guidance
                ),
                seed=request.seed,
                num_images=request.n,
            )
        except (EngineError, EngineUnavailable, ValueError) as exc:
            raise _engine_error(exc) from exc
        data = []
        for image in result.images:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            data.append(
                {"b64_json": base64.b64encode(buffer.getvalue()).decode("ascii"), "revised_prompt": request.prompt}
            )
        return {
            "created": int(time.time()),
            "data": data,
            "seed": result.seed,
            "size": f"{result.width}x{result.height}",
        }

    @app.post("/admin/unload")
    def unload() -> dict:
        engine_manager.close()
        return {"status": "unloaded", "engines": engine_manager.status()}

    return app
