from __future__ import annotations

from local_forge.schemas import ChatMessage
from local_forge.tools import (
    compact_tool_output,
    parse_gemma_tool_calls,
    prepare_openai_tool_messages,
    prepare_tool_messages,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
                "required": ["queries"],
            },
        },
    }
]


def test_parse_native_call():
    calls, content = parse_gemma_tool_calls(
        "call:web_search{queries:[what happened in India today]}", TOOLS
    )
    assert content == ""
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "web_search"
    assert calls[0]["function"]["arguments"] == '{"queries": ["what happened in India today"]}'


def test_parse_call_with_object_argument():
    calls, _ = parse_gemma_tool_calls(
        "call:web_search{queries:{what happened in India today}}", TOOLS
    )
    assert calls[0]["function"]["arguments"] == '{"queries": "what happened in India today"}'


def test_parse_no_call_is_content():
    calls, content = parse_gemma_tool_calls("No tools needed.", TOOLS)
    assert calls == []
    assert content == "No tools needed."


def test_unknown_tool_is_ignored():
    calls, content = parse_gemma_tool_calls("call:other{queries:[x]}", TOOLS)
    assert calls == []
    assert "call:other" in content


JAN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]


def test_query_schema_coerces_queries_list():
    calls, _ = parse_gemma_tool_calls(
        "call:web_search{queries:[what happened in India today]}", JAN_TOOLS
    )
    assert calls[0]["function"]["arguments"] == '{"query": "what happened in India today"}'


def test_query_schema_uses_fallback():
    calls, _ = parse_gemma_tool_calls(
        "call:web_search{}", JAN_TOOLS, fallback_query="India today"
    )
    assert calls[0]["function"]["arguments"] == '{"query": "India today"}'


def test_jan_native_tool_message():
    messages = [
        ChatMessage(
            role="user",
            content=[{"type": "text", "text": {"value": "Find India news"}}],
        ),
        ChatMessage(
            role="assistant",
            content=[
                {
                    "type": "tool_call",
                    "tool_call_id": "call_jan",
                    "tool_name": "web_search",
                    "input": {"query": "India news"},
                    "output": {
                        "kind": "web",
                        "query": "India news",
                        "results": [{"title": "Sample", "text": "Sample result"}],
                    },
                },
                {
                    "type": "text",
                    "text": {"value": "<|tool|>\nweb_search\nquery: India news"},
                },
            ],
        ),
    ]
    converted = prepare_tool_messages(messages, current_date="2026-10-18")
    assert converted[0]["content"] == "Find India news"
    assert converted[1]["tool_calls"][0]["function"]["name"] == "web_search"
    assert converted[2]["tool_responses"][0]["response"]["kind"] == "web"
    assert converted[-1]["role"] == "user"
    assert "<|tool|>" not in str(converted)


def test_compact_tool_output():
    output = {
        "kind": "web",
        "results": [
            {"title": "A", "url": "https://a", "text": "x" * 2000},
            {"title": "B", "url": "https://b", "text": "short"},
        ],
    }
    compact = compact_tool_output(output, max_results=1, max_text_chars=100)
    assert len(compact["results"]) == 1
    assert len(compact["results"][0]["text"]) <= 103
    assert compact["kind"] == "web"


def test_jan_native_to_openai_tool_messages():
    messages = [
        ChatMessage(
            role="user",
            content=[{"type": "text", "text": {"value": "India news?"}}],
        ),
        ChatMessage(
            role="assistant",
            content=[
                {
                    "type": "tool_call",
                    "tool_call_id": "call_jan",
                    "tool_name": "web_search",
                    "input": {"query": "India news"},
                    "output": {"kind": "web", "results": [{"title": "x"}]},
                },
                {"type": "text", "text": {"value": "<|tool|> junk"}},
            ],
        ),
    ]
    converted = prepare_openai_tool_messages(messages)
    assert converted[0] == {"role": "user", "content": "India news?"}
    assert converted[1]["tool_calls"][0]["function"]["name"] == "web_search"
    assert converted[1]["content"] is None
    assert converted[2]["role"] == "tool"
    assert converted[2]["tool_call_id"] == "call_jan"
    assert '"results"' in converted[2]["content"]


def test_new_user_question_after_tool_result():
    messages = [
        ChatMessage(role="user", content="Find news"),
        ChatMessage(
            role="assistant",
            content=[
                {
                    "type": "tool_call",
                    "tool_call_id": "call_jan",
                    "tool_name": "web_search",
                    "input": {"query": "news"},
                    "output": {"results": []},
                }
            ],
        ),
        ChatMessage(role="user", content="Now summarize it differently"),
    ]
    converted = prepare_tool_messages(messages, current_date="2026-10-18")
    assert converted[-1]["content"] == "Now summarize it differently"


def test_prepare_tool_history():
    messages = [
        ChatMessage(role="user", content="Find India news"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"queries": ["India news"]}',
                    },
                }
            ],
        ),
        ChatMessage(
            role="tool",
            tool_call_id="call_1",
            content='{"results": "Sample news"}',
        ),
    ]
    converted = prepare_tool_messages(messages, current_date="2026-10-18")
    assert converted[0]["role"] == "user"
    assert converted[1]["tool_calls"][0]["function"]["name"] == "web_search"
    assert converted[2]["tool_responses"][0]["name"] == "web_search"
    assert converted[-1]["role"] == "user"
    assert "2026-10-18" in converted[-1]["content"]
