from __future__ import annotations

import json
import logging
import sys
import time

from ..catalog import ModelSpec
from ..config import AppConfig
from ..manager import ModelManager
from ..tools import ToolChatResult
from .base import ChatEngine, ChatOptions, EngineError, EngineUnavailable

logger = logging.getLogger(__name__)

try:
    import litert_lm
except ImportError:
    litert_lm = None


def _backend(name: str):
    if litert_lm is None:
        raise EngineUnavailable("litert-lm is not installed. Install it with: pip install litert-lm")
    key = (name or "CPU").upper()
    return getattr(litert_lm.Backend, key, litert_lm.Backend.CPU)


def _proxy_tool(definition: dict):
    class ProxyTool(litert_lm.Tool):
        def __init__(self):
            self.definition = definition

        def get_tool_description(self):
            return self.definition

        def execute(self, param):
            raise NotImplementedError("Tool calls are returned to the client for execution")

    return ProxyTool()


class LiteRTEngine(ChatEngine):
    """First-class LiteRT-LM engine adapter.

    Uses LiteRT's native Conversation API and native chunk.tool_calls only. Bundles whose
    embedded chat template cannot be rendered by the installed runtime are rejected at load.
    No raw-text tool parsing and no template rewriting.
    """

    def __init__(self, spec: ModelSpec, config: AppConfig, manager: ModelManager):
        self.spec = spec
        self.model_id = spec.id
        self.config = config
        self.manager = manager
        self._engine = None
        self._supports_tools = False
        self._load_time = 0.0
        self._backend_name = config.litert_backend

    @property
    def supports_tools(self) -> bool:
        return self._supports_tools

    @property
    def model_path(self):
        return self.manager.path_for(self.spec)

    def load(self) -> None:
        if self._engine is not None:
            return
        if litert_lm is None:
            raise EngineUnavailable("litert-lm is not installed. Install it with: pip install litert-lm")
        path = self.model_path
        if not path.exists() or path.suffix.lower() != ".litertlm":
            raise EngineUnavailable(
                f"LiteRT model must be a local .litertlm file: {path}"
            )
        print(
            f"Loading LiteRT model {self.spec.id} on {self._backend_name} ...",
            file=sys.stderr,
            flush=True,
        )
        started = time.perf_counter()
        try:
            engine = litert_lm.Engine(
                str(path),
                backend=_backend(self._backend_name),
                max_num_tokens=int(self.config.litert_ctx),
            )
            engine.__enter__()
        except Exception as exc:
            raise EngineUnavailable(f"LiteRT could not load {self.spec.id}: {exc}") from exc
        self._engine = engine
        try:
            self._validate_text()
            self._supports_tools = self._probe_tools() if self.config.litert_validate_tools else False
        except EngineUnavailable:
            self.unload()
            raise
        except Exception as exc:
            self.unload()
            raise EngineUnavailable(
                f"LiteRT bundle {self.spec.id} failed validation: {exc}"
            ) from exc
        self._load_time = time.perf_counter() - started
        print(
            f"Loaded {self.spec.id} on {self._backend_name} in {self._load_time:.1f}s "
            f"(native tools: {'yes' if self._supports_tools else 'no'})",
            file=sys.stderr,
            flush=True,
        )

    def _thinking(self):
        return litert_lm.ThinkingConfig(
            enable_thinking=not self.config.disable_thinking,
            thinking_token_budget=0 if self.config.disable_thinking else -1,
        )

    def _sampler(self, options: ChatOptions):
        return litert_lm.SamplerConfig(
            temperature=float(options.temperature),
            top_p=float(options.top_p),
            top_k=int(options.top_k),
        )

    def _validate_text(self) -> None:
        """Reject bundles whose embedded template cannot be rendered by this runtime."""
        conversation = None
        try:
            conversation = self._engine.create_conversation(
                system_message="Validation probe.",
                max_output_tokens=4,
                thinking_config=self._thinking(),
            )
            list(conversation.send_message_async("Hi", max_output_tokens=4))
        except Exception as exc:
            raise EngineUnavailable(
                "LiteRT bundle template is incompatible with the installed runtime. "
                "Use a bundle built for LiteRT-LM 0.18+; do not patch the template. "
                f"Original error: {exc}"
            ) from exc
        finally:
            if conversation is not None:
                try:
                    conversation.close()
                except Exception:
                    pass

    def _probe_tools(self) -> bool:
        """Validate the native tool contract; never parse raw model text."""
        definition = {
            "type": "function",
            "function": {
                "name": "probe_tool",
                "description": "Validation-only tool",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            },
        }
        conversation = None
        calls = []
        try:
            conversation = self._engine.create_conversation(
                tools=[_proxy_tool(definition)],
                automatic_tool_calling=False,
                max_output_tokens=24,
                thinking_config=self._thinking(),
            )
            for chunk in conversation.send_message_async(
                "Call probe_tool with value 'ok'. Do not answer in text.",
                max_output_tokens=24,
            ):
                calls.extend(getattr(chunk, "tool_calls", None) or [])
        except Exception as exc:
            logger.warning("LiteRT tool probe failed: %s", exc)
            return False
        finally:
            if conversation is not None:
                try:
                    conversation.close()
                except Exception:
                    pass
        return bool(calls)

    def _messages(self, messages: list[dict]):
        """Translate OpenAI history into LiteRT's documented message mappings."""
        tool_names: dict[str, str] = {}
        for message in messages:
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                if call.get("id"):
                    tool_names[call["id"]] = function.get("name") or "tool"

        converted = []
        for message in messages:
            role = (message.get("role") or "user").lower()
            content = message.get("content")
            if role == "tool":
                name = message.get("name") or tool_names.get(message.get("tool_call_id") or "") or "tool"
                converted.append(
                    {
                        "role": "tool",
                        "content": [
                            {
                                "type": "tool_response",
                                "name": name,
                                "response": content or "",
                            }
                        ],
                    }
                )
                continue
            if role == "assistant" and message.get("tool_calls"):
                calls = []
                for call in message["tool_calls"]:
                    function = call.get("function") or {}
                    arguments = function.get("arguments") or {}
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    calls.append(
                        {
                            "type": "function",
                            "function": {
                                "name": function.get("name") or "tool",
                                "arguments": arguments,
                            },
                        }
                    )
                converted.append(
                    {"role": "assistant", "content": content, "tool_calls": calls}
                )
                continue
            converted.append({"role": role, "content": content or ""})
        return converted

    def _run(
        self,
        messages: list[dict],
        options: ChatOptions,
        tools: list[dict] | None = None,
    ) -> ToolChatResult:
        self.load()
        if not messages:
            raise ValueError("No messages provided")
        last = messages[-1]
        if (last.get("role") or "").lower() == "tool":
            history = messages
            prompt = "Use the tool result above to answer the original question. Reply concisely."
        else:
            history = messages[:-1]
            prompt = last.get("content") or ""
        tool_objects = [_proxy_tool(tool) for tool in (tools or [])]
        conversation = None
        text_parts: list[str] = []
        native_calls = []
        try:
            conversation = self._engine.create_conversation(
                messages=self._messages(history),
                tools=tool_objects or None,
                automatic_tool_calling=False,
                thinking_config=self._thinking(),
                sampler_config=self._sampler(options),
                max_output_tokens=int(options.max_tokens),
            )
            for chunk in conversation.send_message_async(
                prompt, max_output_tokens=int(options.max_tokens)
            ):
                text_parts.append(str(chunk))
                native_calls.extend(getattr(chunk, "tool_calls", None) or [])
        except Exception as exc:
            raise EngineError(f"LiteRT generation failed: {exc}") from exc
        finally:
            if conversation is not None:
                try:
                    conversation.close()
                except Exception:
                    pass
        openai_calls = [
            {
                "id": call.id or f"call_{index}",
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(dict(call.arguments), ensure_ascii=False),
                },
            }
            for index, call in enumerate(native_calls)
        ]
        return ToolChatResult(
            content="".join(text_parts).strip(),
            tool_calls=openai_calls,
            raw_text="".join(text_parts),
        )

    def chat(self, messages: list[dict], images: list[bytes], options: ChatOptions) -> str:
        if images:
            raise EngineError("The LiteRT adapter does not implement image input yet")
        return self._run(messages, options).content

    def chat_tools(
        self,
        messages: list[dict],
        images: list[bytes],
        options: ChatOptions,
        tools: list[dict] | None = None,
    ) -> ToolChatResult:
        if not self._supports_tools:
            raise EngineError(
                "This LiteRT bundle did not pass the native tool-call validation. "
                "Use a compatible bundle; raw tool text is not parsed."
            )
        if images:
            raise EngineError("The LiteRT adapter does not implement image input yet")
        return self._run(messages, options, tools or options.tools)

    def stream_chat(self, messages: list[dict], images: list[bytes], options: ChatOptions):
        if images:
            raise EngineError("The LiteRT adapter does not implement image input yet")
        result = self._run(messages, options)
        if result.content:
            for start in range(0, len(result.content), 48):
                yield result.content[start : start + 48]

    def unload(self) -> None:
        engine = self._engine
        self._engine = None
        self._supports_tools = False
        if engine is not None:
            try:
                engine.__exit__(None, None, None)
            except Exception:
                pass
        print(f"Unloaded {self.spec.id}", file=sys.stderr, flush=True)

    def describe(self) -> dict:
        return {
            "id": self.spec.id,
            "engine": "litert",
            "model_path": str(self.model_path),
            "backend": self._backend_name,
            "loaded": self._engine is not None,
            "supports_tools": self._supports_tools,
            "load_time_s": round(self._load_time, 1),
            "context": self.config.litert_ctx,
        }
