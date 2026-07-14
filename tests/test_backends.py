from kultivait.backends import (
    LlamaCppBackend,
    OllamaBackend,
    from_ollama_tool_calls,
    is_truncated,
    merge_tool_call_deltas,
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


def test_llamacpp_payload_is_openai_native():
    # llama-server speaks OpenAI format directly: no message translation,
    # no options.num_ctx (context size is fixed at server launch via -c).
    backend = LlamaCppBackend("qwen2.5-14b-instruct-q4_k_m.gguf")
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "bash"}}]
    payload = backend._payload(messages, tools, stream=True)
    assert payload["model"] == "qwen2.5-14b-instruct-q4_k_m.gguf"
    assert payload["messages"] == messages
    assert payload["tools"] == tools
    assert payload["stream"] is True
    assert "options" not in payload


def test_llamacpp_parses_openai_response_into_completion():
    data = {
        "choices": [
            {
                "message": {
                    "content": "done",
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"cmd": "ls"}'},
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }
    completion = LlamaCppBackend._parse(data)
    assert completion.text == "done"
    assert completion.tokens_in == 12
    assert completion.tokens_out == 3
    assert completion.cost_usd == 0.0
    assert completion.local is True
    assert completion.tool_calls[0]["function"]["name"] == "bash"
    # truncation detection is an ollama quirk (pinned prompt_eval_count);
    # llama.cpp has no equivalent signal, so this stays False
    assert completion.truncated is False


def test_merge_tool_call_deltas_accumulates_streamed_fragments():
    # OpenAI streaming splits a tool call across chunks: the first carries
    # id/name, later ones append argument text, keyed by index.
    acc: dict = {}
    merge_tool_call_deltas(
        acc, [{"index": 0, "id": "call_1", "function": {"name": "bash", "arguments": '{"cm'}}]
    )
    merge_tool_call_deltas(acc, [{"index": 0, "function": {"arguments": 'd": "ls"}'}}])
    assert acc[0]["id"] == "call_1"
    assert acc[0]["function"]["name"] == "bash"
    assert acc[0]["function"]["arguments"] == '{"cmd": "ls"}'


def test_ollama_tool_calls_convert_to_openai_format():
    ollama_calls = [{"function": {"name": "bash", "arguments": {"cmd": "ls"}}}]
    converted = from_ollama_tool_calls(ollama_calls)
    assert len(converted) == 1
    call = converted[0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "bash"
    assert call["function"]["arguments"] == '{"cmd": "ls"}'  # JSON string
    assert call["id"].startswith("call_")
