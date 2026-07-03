from kultivait.escalations import EscalationStore, render_transcript


MESSAGES = [
    {"role": "user", "content": "draft a technical spec for PDF reports"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read", "arguments": '{"path": "docs/spec.md"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "ENOENT"},
    {"role": "assistant", "content": "The file does not exist."},
]


def test_store_roundtrip_and_listing(tmp_path):
    store = EscalationStore(tmp_path)
    eid = store.save(MESSAGES, requested_tier="claude")
    listed = store.list()
    assert len(listed) == 1
    assert listed[0].id == eid
    assert listed[0].requested_tier == "claude"
    assert listed[0].snippet.startswith("draft a technical spec")
    assert store.load_messages(eid) == MESSAGES


def test_list_is_newest_last_across_instances(tmp_path):
    EscalationStore(tmp_path).save(MESSAGES, requested_tier="claude")
    store = EscalationStore(tmp_path)
    second = store.save(MESSAGES, requested_tier="gemini:agy")
    listed = store.list()
    assert len(listed) == 2
    assert listed[-1].id == second


def test_render_transcript_covers_tools_and_roles():
    text = render_transcript(MESSAGES)
    assert "[user] draft a technical spec for PDF reports" in text
    assert 'read({"path": "docs/spec.md"})' in text
    assert "[tool] ENOENT" in text
    assert "[assistant] The file does not exist." in text
