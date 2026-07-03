"""OpenAI-compatible proxy: weigh locally, route deliberately, tally everything."""

import json
import time
import uuid
from typing import Callable

import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from kultivait.backends import Backend, Completion
from kultivait.gates import Gate
from kultivait.ledger import Ledger
from kultivait.router import Decision, Router


def _text_of(content) -> str:
    """Message content may be a string or a list of content blocks/parts."""
    if isinstance(content, str):
        return content
    return " ".join(
        block.get("text", "") for block in content if isinstance(block, dict)
    )


def _normalize(messages: list[dict]) -> list[dict]:
    """Flatten content blocks/parts to plain strings: backends (ollama, CLIs)
    understand neither Anthropic blocks nor OpenAI content parts.
    Tool plumbing (assistant tool_calls, tool results) is preserved."""
    out = []
    for m in messages:
        norm = {"role": m.get("role", "user"), "content": _text_of(m.get("content") or "")}
        if m.get("tool_calls"):
            norm["tool_calls"] = m["tool_calls"]
        if m.get("tool_call_id"):
            norm["tool_call_id"] = m["tool_call_id"]
        out.append(norm)
    return out


def create_app(
    router: Router,
    embed: Callable[[str], np.ndarray],
    backends: dict[str, Backend],
    ledger: Ledger,
    gate: Gate,
) -> FastAPI:
    app = FastAPI(title="kultivait")

    def _record(tier: str, completion: Completion, **decision_meta) -> None:
        ledger.record(
            tier=tier,
            local=completion.local,
            tokens_in=completion.tokens_in,
            tokens_out=completion.tokens_out,
            cost_usd=completion.cost_usd,
            truncated=completion.truncated,
            **decision_meta,
        )

    def _decision_meta(decision: Decision, tool_fallback: bool, messages: list[dict]) -> dict:
        user_text = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        return {
            "requested_tier": decision.tier,
            "margin": round(decision.margin, 4),
            "escalated": decision.escalated,
            "tool_fallback": tool_fallback,
            "snippet": user_text[:80],
        }

    def _classify(messages: list[dict]) -> "Decision":
        user_text = next(
            (_text_of(m["content"]) for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        return router.classify(embed(user_text))

    def _tool_capable_tier(tier: str) -> tuple[str, bool]:
        """Cloud CLI backends run their own agent loops and can't return
        client-side tool calls; tools-bearing requests fall back to the most
        capable local tier that supports tools."""
        if backends[tier].supports_tools:
            return tier, False
        for name in reversed(router.capability_order):
            if backends[name].supports_tools:
                return name, True
        raise RuntimeError("no tool-capable backend configured")

    @app.post("/v1/chat/completions")
    def chat_completions(body: dict):
        messages = _normalize(body.get("messages", []))
        tools = body.get("tools")
        decision = _classify(messages)
        tier, tool_fallback = (
            _tool_capable_tier(decision.tier) if tools else (decision.tier, False)
        )
        meta = _decision_meta(decision, tool_fallback, messages)

        def kultivait_meta(local: bool) -> dict:
            return {
                "tier": tier,
                "margin": decision.margin,
                "escalated": decision.escalated,
                "tool_fallback": tool_fallback,
                "local": local,
            }

        if body.get("stream"):
            chunk_id = f"kult-{uuid.uuid4().hex[:12]}"
            created = int(time.time())

            def chunk(delta: dict, finish: str | None = None) -> str:
                payload = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": tier,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                return f"data: {json.dumps(payload)}\n\n"

            def sse():
                yield chunk({"role": "assistant"})
                for item in backends[tier].stream(messages, tools=tools):
                    if isinstance(item, Completion):
                        _record(tier, item, **meta)
                        if item.tool_calls:
                            yield chunk(
                                {
                                    "tool_calls": [
                                        {**tc, "index": i}
                                        for i, tc in enumerate(item.tool_calls)
                                    ]
                                }
                            )
                            yield chunk({}, finish="tool_calls")
                        else:
                            yield chunk({}, finish="stop")
                    else:
                        yield chunk({"content": item})
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        completion = backends[tier].complete(messages, tools=tools)
        _record(tier, completion, **meta)
        message: dict = {"role": "assistant", "content": completion.text or None}
        if completion.tool_calls:
            message["tool_calls"] = completion.tool_calls
        return {
            "id": f"kult-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": tier,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if completion.tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": completion.tokens_in,
                "completion_tokens": completion.tokens_out,
                "total_tokens": completion.tokens_in + completion.tokens_out,
            },
            "kultivait": kultivait_meta(completion.local),
        }

    @app.post("/v1/messages")
    def anthropic_messages(body: dict):
        messages = _normalize(body.get("messages", []))
        system = body.get("system")
        if system:
            messages = [{"role": "system", "content": _text_of(system)}, *messages]
        decision = _classify(messages)
        meta = _decision_meta(decision, False, messages)
        msg_id = f"kult-{uuid.uuid4().hex[:12]}"

        if body.get("stream"):

            def event(etype: str, payload: dict) -> str:
                return f"event: {etype}\ndata: {json.dumps({'type': etype, **payload})}\n\n"

            def sse():
                # input token count isn't known until the backend finishes;
                # real usage arrives in message_delta, per-field zeros here.
                yield event(
                    "message_start",
                    {
                        "message": {
                            "id": msg_id,
                            "type": "message",
                            "role": "assistant",
                            "model": decision.tier,
                            "content": [],
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        }
                    },
                )
                yield event(
                    "content_block_start",
                    {"index": 0, "content_block": {"type": "text", "text": ""}},
                )
                for item in backends[decision.tier].stream(messages):
                    if isinstance(item, Completion):
                        _record(decision.tier, item, **meta)
                        yield event("content_block_stop", {"index": 0})
                        yield event(
                            "message_delta",
                            {
                                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                "usage": {
                                    "input_tokens": item.tokens_in,
                                    "output_tokens": item.tokens_out,
                                },
                            },
                        )
                    else:
                        yield event(
                            "content_block_delta",
                            {"index": 0, "delta": {"type": "text_delta", "text": item}},
                        )
                yield event("message_stop", {})

            return StreamingResponse(sse(), media_type="text/event-stream")

        completion = backends[decision.tier].complete(messages)
        _record(decision.tier, completion, **meta)
        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": decision.tier,
            "content": [{"type": "text", "text": completion.text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": completion.tokens_in,
                "output_tokens": completion.tokens_out,
            },
        }

    @app.post("/gate")
    def gate_handoff(body: dict):
        result = gate.distill(
            body["transcript"],
            from_phase=body.get("from_phase", "previous"),
            to_phase=body.get("to_phase", "next"),
        )
        return {
            "brief": result.brief,
            "tokens_before": result.tokens_before,
            "tokens_after": result.tokens_after,
            "compost_id": result.compost_id,
        }

    @app.get("/harvest")
    def harvest():
        return ledger.harvest()

    return app
