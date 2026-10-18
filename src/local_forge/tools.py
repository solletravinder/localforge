from __future__ import annotations

import copy
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .schemas import ChatMessage

_CALL_RE = re.compile(r"call:([A-Za-z0-9_.-]+)\s*")


@dataclass
class ParsedToolCall:
    id: str
    name: str
    arguments: Any


@dataclass
class ToolChatResult:
    content: str
    tool_calls: list[dict]
    raw_text: str = ""
    model_id: str = ""


def _skip_ws(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _parse_string(text: str, index: int) -> tuple[str, int]:
    if text.startswith('<|"|>', index):
        end = text.find('<|"|>', index + 5)
        if end < 0:
            return text[index + 5 :], len(text)
        return text[index + 5 : end], end + 5
    if index < len(text) and text[index] == '"':
        try:
            value, end = json.JSONDecoder().raw_decode(text[index:])
            return str(value), index + end
        except json.JSONDecodeError:
            pass
    return "", index


def _parse_bare(text: str, index: int) -> tuple[Any, int]:
    start = index
    while index < len(text) and text[index] not in ",}]":
        index += 1
    raw = text[start:index].strip()
    lowered = raw.lower()
    if lowered == "true":
        return True, index
    if lowered == "false":
        return False, index
    if lowered == "null":
        return None, index
    try:
        return int(raw), index
    except ValueError:
        pass
    try:
        return float(raw), index
    except ValueError:
        pass
    return raw, index


def _parse_value(text: str, index: int) -> tuple[Any, int]:
    index = _skip_ws(text, index)
    if index >= len(text):
        return "", index
    if text[index] == "{":
        return _parse_object(text, index)
    if text[index] == "[":
        return _parse_list(text, index)
    if text[index] == '"' or text.startswith('<|"|>', index):
        return _parse_string(text, index)
    return _parse_bare(text, index)


def _parse_object(text: str, index: int) -> tuple[Any, int]:
    index = _skip_ws(text, index)
    if index >= len(text) or text[index] != "{":
        return _parse_top_pairs(text, index)[0]
    index += 1
    index = _skip_ws(text, index)
    if index < len(text) and text[index] == "}":
        return {}, index + 1
    first = True
    result: dict[str, Any] = {}
    while index < len(text):
        index = _skip_ws(text, index)
        if index < len(text) and text[index] == "}":
            return (result if result else text[text.find("{") + 1 : index].strip()), index + 1
        key, next_index = _parse_key(text, index)
        next_index = _skip_ws(text, next_index)
        if next_index >= len(text) or text[next_index] != ":":
            # Gemma occasionally emits an object containing only a bare value.
            end = text.find("}", index)
            if end < 0:
                return text[index:].strip(), len(text)
            return text[index:end].strip(), end + 1
        value, next_index = _parse_value(text, next_index + 1)
        result[key] = value
        index = _skip_ws(text, next_index)
        if index < len(text) and text[index] == ",":
            index += 1
            first = False
            continue
        if index < len(text) and text[index] == "}":
            return result, index + 1
        break
    _ = first
    return result, index


def _parse_list(text: str, index: int) -> tuple[list[Any], int]:
    index += 1
    result: list[Any] = []
    while index < len(text):
        index = _skip_ws(text, index)
        if index < len(text) and text[index] == "]":
            return result, index + 1
        value, index = _parse_value(text, index)
        result.append(value)
        index = _skip_ws(text, index)
        if index < len(text) and text[index] == ",":
            index += 1
            continue
        if index < len(text) and text[index] == "]":
            return result, index + 1
        break
    return result, index


def _parse_key(text: str, index: int) -> tuple[str, int]:
    if text[index] == '"' or text.startswith('<|"|>', index):
        value, end = _parse_string(text, index)
        return value, end
    start = index
    while index < len(text) and text[index] not in ":,}":
        index += 1
    return text[start:index].strip(), index


def _parse_top_pairs(text: str, index: int = 0) -> tuple[Any, int]:
    index = _skip_ws(text, index)
    if index >= len(text):
        return {}, index
    if text[index] == "{":
        return _parse_object(text, index)
    result: dict[str, Any] = {}
    while index < len(text):
        key, next_index = _parse_key(text, index)
        next_index = _skip_ws(text, next_index)
        if next_index >= len(text) or text[next_index] != ":":
            return text[index:].strip(), len(text)
        value, next_index = _parse_value(text, next_index + 1)
        result[key] = value
        index = _skip_ws(text, next_index)
        if index < len(text) and text[index] == ",":
            index += 1
            continue
        break
    return result, index


def _parse_arguments(body: str) -> Any:
    body = body.strip()
    if not body:
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    try:
        parsed, _ = _parse_top_pairs(body)
        return parsed
    except Exception:
        return body


def _find_call(text: str, start: int) -> tuple[str, str, int] | None:
    match = _CALL_RE.search(text, start)
    if not match:
        return None
    name = match.group(1)
    brace = text.find("{", match.end())
    if brace < 0:
        return None
    depth = 0
    index = brace
    in_quote = False
    in_special = False
    while index < len(text):
        if text.startswith('<|"|>', index):
            in_special = not in_special
            index += 5
            continue
        char = text[index]
        if not in_special:
            if char == '"' and (index == 0 or text[index - 1] != "\\"):
                in_quote = not in_quote
            elif not in_quote:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return name, text[brace + 1 : index], index + 1
        index += 1
    return None


def _tool_schema(tools: list[dict] | None, name: str) -> dict | None:
    for tool in tools or []:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        if function.get("name") == name:
            return function
    return None


def sanitize_tools_for_gemma(tools: list[dict] | None) -> list[dict]:
    """Normalize tool schemas for Gemma's Minja template.

    Minja cannot render union types like ["string", "null"], which many OpenAI clients emit.
    Flatten unions, force a single lowercase type, ensure descriptions, and make enums lists.
    """
    cleaned = copy.deepcopy(tools or [])

    def normalize(schema: dict) -> None:
        if not isinstance(schema, dict):
            return
        for union_key in ("anyOf", "oneOf", "allOf"):
            options = schema.get(union_key)
            if isinstance(options, list) and options:
                candidates = [
                    option
                    for option in options
                    if isinstance(option, dict) and option.get("type") != "null"
                ]
                if candidates:
                    merged = copy.deepcopy(candidates[0])
                    for key, value in schema.items():
                        if key not in {"anyOf", "oneOf", "allOf"}:
                            merged.setdefault(key, value)
                    schema.clear()
                    schema.update(merged)
                break
        raw_type = schema.get("type")
        nullable = bool(schema.get("nullable"))
        if isinstance(raw_type, list):
            non_null = [item for item in raw_type if item not in (None, "null")]
            nullable = nullable or "null" in raw_type
            raw_type = non_null[0] if non_null else "string"
        if raw_type is None:
            if isinstance(schema.get("properties"), dict):
                raw_type = "object"
            elif isinstance(schema.get("items"), dict):
                raw_type = "array"
            else:
                raw_type = "string"
        schema["type"] = str(raw_type).lower()
        if nullable:
            schema["nullable"] = True
        if "enum" in schema and not isinstance(schema["enum"], list):
            schema["enum"] = [schema["enum"]]
        description = schema.get("description")
        if description is not None and not isinstance(description, str):
            schema["description"] = str(description)
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for name, prop in properties.items():
                if isinstance(prop, dict):
                    prop.setdefault("description", str(name).replace("_", " "))
                    normalize(prop)
        items = schema.get("items")
        if isinstance(items, dict):
            items.setdefault("description", "item")
            normalize(items)

    for tool in cleaned:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        if not isinstance(function, dict):
            continue
        function.setdefault("description", function.get("name", "tool"))
        parameters = function.get("parameters")
        if isinstance(parameters, dict):
            parameters.setdefault("description", "Tool parameters")
            normalize(parameters)
    return cleaned


def _coerce_arguments(
    arguments: Any, tool: dict | None, fallback_query: str | None = None
) -> Any:
    parameters = (tool or {}).get("parameters") or {}
    properties = parameters.get("properties") or {}
    required = parameters.get("required") or list(properties)

    if "query" in properties:
        query_value: Any = None
        if isinstance(arguments, dict):
            candidate = arguments.get("query")
            if candidate not in (None, "", [], {}):
                query_value = candidate
            if query_value is None:
                for alias in ("queries", "q", "search_query", "searchQuery", "text", "input"):
                    candidate = arguments.get(alias)
                    if candidate not in (None, "", [], {}):
                        query_value = candidate
                        break
            if query_value is None:
                for candidate in arguments.values():
                    if isinstance(candidate, str) and candidate.strip():
                        query_value = candidate
                        break
                    if isinstance(candidate, list) and candidate and isinstance(candidate[0], str):
                        query_value = candidate[0]
                        break
        elif isinstance(arguments, list) and arguments:
            query_value = arguments[0]
        elif isinstance(arguments, str):
            query_value = arguments
        if isinstance(query_value, dict):
            query_value = query_value.get("query") or next(
                (value for value in query_value.values() if isinstance(value, str)), ""
            )
        if isinstance(query_value, list):
            query_value = next(
                (value for value in query_value if isinstance(value, str)), ""
            )
        if not query_value:
            query_value = fallback_query
        if query_value:
            filtered = (
                {key: value for key, value in arguments.items() if key in properties}
                if isinstance(arguments, dict)
                else {}
            )
            filtered["query"] = str(query_value)
            return filtered

    if not isinstance(arguments, (str, list)):
        return arguments
    if len(required) == 1:
        return {required[0]: arguments}
    if "queries" in properties:
        return {"queries": arguments if isinstance(arguments, list) else [arguments]}
    if "query" in properties:
        return {"query": arguments[0] if isinstance(arguments, list) and arguments else arguments}
    return {"input": arguments}


def parse_gemma_tool_calls(
    text: str,
    tools: list[dict] | None = None,
    fallback_query: str | None = None,
) -> tuple[list[dict], str]:
    """Parse Gemma's native tool call syntax into OpenAI tool_calls.

    Returns (tool_calls, content_without_calls).
    """
    cleaned = (
        text.replace("<|tool_call>", "")
        .replace("<tool_call|>", "")
        .replace("<|tool_response>", "")
        .replace("<tool_response|>", "")
    )
    allowed = {
        tool.get("function", {}).get("name")
        for tool in (tools or [])
        if isinstance(tool, dict)
    }
    calls: list[dict] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        found = _find_call(cleaned, cursor)
        if not found:
            break
        name, body, end = found
        start = cleaned.rfind("call:", cursor, end)
        cursor = end
        if allowed and name not in allowed:
            continue
        function = _tool_schema(tools, name)
        arguments = _coerce_arguments(
            _parse_arguments(body), function, fallback_query=fallback_query
        )
        calls.append(
            {
                "id": "call_" + uuid.uuid4().hex[:16],
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )
        if start >= 0:
            spans.append((start, end))
    content = cleaned
    for start, end in reversed(spans):
        content = content[:start] + content[end:]
    content = content.strip()
    return calls, content


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "input_text"}:
                value = item.get("text") or item.get("value")
                if isinstance(value, dict):
                    value = value.get("value")
                if value:
                    parts.append(str(value))
        return "\n".join(parts)
    return ""


def compact_tool_output(
    output: Any, max_results: int = 5, max_text_chars: int = 600
) -> Any:
    """Trim large web-search payloads to titles/URLs plus short snippets.

    Exa returns full page text. On a CPU/hybrid model that dominates prefill time and memory.
    Jan still shows the full sources; the model only needs the headline and lead.
    """
    if not isinstance(output, dict) or not isinstance(output.get("results"), list):
        return output
    compact = {key: value for key, value in output.items() if key != "results"}
    results = []
    for item in output["results"][:max_results]:
        if not isinstance(item, dict):
            results.append(item)
            continue
        entry = {key: value for key, value in item.items() if key != "text"}
        if isinstance(item.get("text"), str):
            text = item["text"]
            entry["text"] = (
                text[:max_text_chars].rstrip() + "..."
                if len(text) > max_text_chars
                else text
            )
        results.append(entry)
    compact["results"] = results
    return compact


def _tool_call_parts(message: ChatMessage) -> list[dict]:
    content = message.content
    if not isinstance(content, list):
        return []
    return [
        item
        for item in content
        if isinstance(item, dict) and item.get("type") == "tool_call"
    ]


def _extra(message: ChatMessage, key: str) -> Any:
    return (message.model_extra or {}).get(key)


def prepare_openai_tool_messages(messages: list[ChatMessage]) -> list[dict]:
    """Convert Jan's native tool-call content into standard OpenAI messages.

    llama.cpp speaks the OpenAI tool format, not Jan's internal content parts. This keeps both
    standard clients and Jan working through the same proxy.
    """
    converted: list[dict] = []
    for message in messages:
        role = (message.role or "").strip().lower()
        if role == "developer":
            role = "system"
        text = _content_text(message.content)
        native_calls = _tool_call_parts(message) if role == "assistant" else []
        if native_calls:
            calls = []
            tool_results = []
            for part in native_calls:
                name = part.get("tool_name") or part.get("name") or "tool"
                call_id = part.get("tool_call_id") or "call_" + uuid.uuid4().hex[:16]
                arguments = part.get("input")
                if arguments is None:
                    arguments = part.get("arguments")
                if arguments is None:
                    arguments = {}
                calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(arguments, ensure_ascii=False)
                            if not isinstance(arguments, str)
                            else arguments,
                        },
                    }
                )
                output = part.get("output")
                if output is not None:
                    output = compact_tool_output(output)
                    content = (
                        output
                        if isinstance(output, str)
                        else json.dumps(output, ensure_ascii=False)
                    )
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": name,
                            "content": content,
                        }
                    )
            converted.append(
                {"role": "assistant", "content": None, "tool_calls": calls}
            )
            converted.extend(tool_results)
            continue
        if role == "tool":
            item: dict = {"role": "tool", "content": text}
            call_id = _extra(message, "tool_call_id")
            name = _extra(message, "name")
            if call_id:
                item["tool_call_id"] = call_id
            if name:
                item["name"] = name
            converted.append(item)
            continue
        raw_calls = _extra(message, "tool_calls")
        if role == "assistant" and raw_calls:
            calls = []
            for call in raw_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                arguments = function.get("arguments")
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                calls.append(
                    {
                        "id": call.get("id") or "call_" + uuid.uuid4().hex[:16],
                        "type": "function",
                        "function": {
                            "name": function.get("name") or "tool",
                            "arguments": arguments or "{}",
                        },
                    }
                )
            if calls:
                converted.append(
                    {"role": "assistant", "content": text or None, "tool_calls": calls}
                )
                continue
        converted.append({"role": role, "content": text})
    return converted


def prepare_tool_messages(
    messages: list[ChatMessage], current_date: str | None = None
) -> list[dict]:
    """Convert OpenAI tool history to Gemma's template message format."""
    converted: list[dict] = []
    tool_names: dict[str, str] = {}
    pending_responses: list[dict] = []
    last_was_tool = False

    def flush() -> None:
        nonlocal pending_responses, last_was_tool
        if pending_responses:
            converted.append(
                {"role": "tool", "content": "", "tool_responses": pending_responses}
            )
            pending_responses = []
            last_was_tool = True

    for message in messages:
        role = (message.role or "").strip().lower()
        if role == "developer":
            role = "system"
        text = _content_text(message.content)
        native_calls = _tool_call_parts(message) if role == "assistant" else []
        if native_calls:
            flush()
            calls = []
            responses = []
            for part in native_calls:
                name = part.get("tool_name") or part.get("name") or "tool"
                call_id = part.get("tool_call_id") or "call_" + uuid.uuid4().hex[:16]
                arguments = part.get("input")
                if arguments is None:
                    arguments = part.get("arguments")
                if arguments is None:
                    arguments = {}
                tool_names[call_id] = name
                calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                )
                output = part.get("output")
                if output is not None:
                    responses.append(
                        {"name": name, "response": compact_tool_output(output)}
                    )
            converted.append({"role": "assistant", "content": "", "tool_calls": calls})
            if responses:
                converted.append(
                    {"role": "tool", "content": "", "tool_responses": responses}
                )
                last_was_tool = True
            else:
                last_was_tool = False
            continue
        if role == "tool":
            name = _extra(message, "name")
            call_id = _extra(message, "tool_call_id")
            if not name and call_id:
                name = tool_names.get(call_id)
            name = name or "tool"
            response: Any = text
            try:
                response = json.loads(text) if text else {}
            except json.JSONDecodeError:
                pass
            pending_responses.append({"name": name, "response": response})
            continue
        flush()
        if role == "assistant":
            raw_calls = _extra(message, "tool_calls") or []
            calls = []
            for call in raw_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                name = function.get("name") or "tool"
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
                call_id = call.get("id") or "call_" + uuid.uuid4().hex[:16]
                tool_names[call_id] = name
                calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                )
            item: dict = {"role": "assistant", "content": text}
            if calls:
                item["tool_calls"] = calls
            converted.append(item)
            last_was_tool = False
        else:
            converted.append({"role": role, "content": text})
            last_was_tool = False
    flush()

    if last_was_tool and converted:
        date_text = current_date or datetime.now().strftime("%Y-%m-%d")
        converted.append(
            {
                "role": "user",
                "content": (
                    "Use the tool result above to answer the user's original question. "
                    f"The current local date is {date_text}. Do not call the same tool again "
                    "unless the result was insufficient."
                ),
            }
        )
    return converted
