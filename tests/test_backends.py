from kultivait.backends import (
    OllamaBackend,
    from_ollama_tool_calls,
    is_truncated,
    to_ollama_messages,
)


def test_truncation_detected_at_context_boundary():
    # ollama truncates to num_ctx - 1 and reports that as prompt_eval_count
    # (observed live: limit=8191 for the 8192 default, limit=32767 for 32768)
    assert is_truncated(prompt_eval_count=8191, num_ctx=8192) is True
    assert is_truncated(prompt_eval_count=32767, num_ctx=32768) is True
    assert is_truncated(prompt_eval_count=5000, num_ctx=8192) is False


def test_payload_sets_num_ctx_to_avoid_input_truncation():
    # ollama defaults num_ctx to 2048/8192 and silently truncates longer
    # prompts; agent clients (Pi) send envelopes well past that.
    backend = OllamaBackend("qwen3:14b", num_ctx=32768)
    payload = backend._payload([{"role": "user", "content": "hi"}], None, stream=False)
    assert payload["options"]["num_ctx"] == 32768


def test_num_ctx_defaults_to_a_generous_window():
    backend = OllamaBackend("qwen3:14b")
    payload = backend._payload([{"role": "user", "content": "hi"}], None, stream=False)
    assert payload["options"]["num_ctx"] >= 32768


def test_openai_tool_history_converts_to_ollama_format():
    messages = [
        {"role": "user", "content": "read a.py"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path": "a.py"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "print('hi')"},
    ]
    converted = to_ollama_messages(messages)
    # ollama wants dict arguments and knows nothing of OpenAI ids
    assert converted[1]["tool_calls"] == [
        {"function": {"name": "read", "arguments": {"path": "a.py"}}}
    ]
    assert converted[2] == {"role": "tool", "content": "print('hi')"}
    # plain messages pass through untouched
    assert converted[0] == {"role": "user", "content": "read a.py"}


def test_ollama_tool_calls_convert_to_openai_format():
    ollama_calls = [{"function": {"name": "bash", "arguments": {"cmd": "ls"}}}]
    converted = from_ollama_tool_calls(ollama_calls)
    assert len(converted) == 1
    call = converted[0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "bash"
    assert call["function"]["arguments"] == '{"cmd": "ls"}'  # JSON string
    assert call["id"].startswith("call_")
