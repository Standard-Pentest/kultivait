"""Model backends: local ollama and cloud CLIs, behind one interface.

`stream()` yields text deltas and finishes with a Completion carrying the
final usage, so callers can tally the ledger after the stream ends.
"""

import json
from dataclasses import dataclass
from typing import Iterator, Protocol


@dataclass(frozen=True)
class Completion:
    text: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    local: bool
    tool_calls: "list[dict] | None" = None
    truncated: bool = False


def is_truncated(prompt_eval_count: int, num_ctx: int) -> bool:
    """ollama silently clips over-long prompts to num_ctx - 1 tokens, keeping
    the tail and amputating the head (system prompts, skill instructions).
    A prompt_eval_count pinned at the boundary is the tell."""
    return prompt_eval_count >= num_ctx - 1


class Backend(Protocol):
    supports_tools: bool

    def complete(self, messages: list[dict], tools: "list[dict] | None" = None) -> Completion: ...

    def stream(
        self, messages: list[dict], tools: "list[dict] | None" = None
    ) -> Iterator["str | Completion"]: ...


def to_ollama_messages(messages: list[dict]) -> list[dict]:
    """OpenAI-format history -> ollama native format: tool-call arguments
    become dicts, and OpenAI's id plumbing (ids, tool_call_id) is dropped."""
    out = []
    for m in messages:
        norm = {"role": m.get("role", "user"), "content": m.get("content") or ""}
        if m.get("tool_calls"):
            norm["tool_calls"] = [
                {
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": json.loads(tc["function"]["arguments"])
                        if isinstance(tc["function"].get("arguments"), str)
                        else tc["function"].get("arguments", {}),
                    }
                }
                for tc in m["tool_calls"]
            ]
        out.append(norm)
    return out


def from_ollama_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """ollama tool calls -> OpenAI format: dict arguments become JSON strings,
    and ids are generated (ollama doesn't issue them)."""
    import uuid

    return [
        {
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": tc["function"]["name"],
                "arguments": json.dumps(tc["function"].get("arguments", {})),
            },
        }
        for tc in tool_calls
    ]


class OllamaBackend:
    """Local model via the ollama chat API. Free by definition."""

    supports_tools = True

    def __init__(
        self, model: str, base_url: str = "http://localhost:11434", num_ctx: int = 32768
    ):
        self.model = model
        self.base_url = base_url
        self.num_ctx = num_ctx

    def _payload(self, messages: list[dict], tools: "list[dict] | None", stream: bool) -> dict:
        # ollama defaults to a small context and silently truncates longer
        # prompts; agent clients send envelopes well past the default, so
        # classification would run on a prompt the model never fully sees.
        payload = {
            "model": self.model,
            "messages": to_ollama_messages(messages),
            "stream": stream,
            "options": {"num_ctx": self.num_ctx},
        }
        if tools:
            payload["tools"] = tools
        return payload

    def complete(self, messages: list[dict], tools: "list[dict] | None" = None) -> Completion:
        import httpx

        r = httpx.post(
            f"{self.base_url}/api/chat",
            json=self._payload(messages, tools, stream=False),
            timeout=300,
        )
        r.raise_for_status()
        data = r.json()
        raw_calls = data["message"].get("tool_calls") or []
        tokens_in = data.get("prompt_eval_count", 0)
        return Completion(
            text=data["message"]["content"],
            tokens_in=tokens_in,
            tokens_out=data.get("eval_count", 0),
            cost_usd=0.0,
            local=True,
            tool_calls=from_ollama_tool_calls(raw_calls) if raw_calls else None,
            truncated=is_truncated(tokens_in, self.num_ctx),
        )

    def stream(
        self, messages: list[dict], tools: "list[dict] | None" = None
    ) -> Iterator["str | Completion"]:
        import httpx

        parts: list[str] = []
        raw_calls: list[dict] = []
        with httpx.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=self._payload(messages, tools, stream=True),
            timeout=300,
        ) as r:
            r.raise_for_status()
            data = {}
            for line in r.iter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                message = data.get("message", {})
                raw_calls.extend(message.get("tool_calls") or [])
                delta = message.get("content", "")
                if delta:
                    parts.append(delta)
                    yield delta
        tokens_in = data.get("prompt_eval_count", 0)
        yield Completion(
            text="".join(parts),
            tokens_in=tokens_in,
            tokens_out=data.get("eval_count", 0),
            cost_usd=0.0,
            local=True,
            tool_calls=from_ollama_tool_calls(raw_calls) if raw_calls else None,
            truncated=is_truncated(tokens_in, self.num_ctx),
        )


class CLIBackend:
    """Cloud model behind a print-mode CLI (`claude -p`, `agy -p`).

    CLIs don't report token usage, so tokens are estimated at ~4 chars/token
    and cost from the configured per-million pricing. They run their own
    agent loops, so client-side tool calls are unsupported.
    """

    supports_tools = False

    def __init__(self, command: list[str], price_in: float, price_out: float):
        self.command = command
        self.price_in = price_in
        self.price_out = price_out

    def complete(self, messages: list[dict], tools: "list[dict] | None" = None) -> Completion:
        import subprocess

        prompt = "\n\n".join(
            f"[{m.get('role', 'user')}] {m.get('content', '')}" for m in messages
        )
        result = subprocess.run(
            [*self.command, "-p", prompt],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{self.command[0]} exited {result.returncode}: {result.stderr.strip()[:500]}"
            )
        text = result.stdout.strip()
        tokens_in = max(1, len(prompt) // 4)
        tokens_out = max(1, len(text) // 4)
        cost = (tokens_in * self.price_in + tokens_out * self.price_out) / 1e6
        return Completion(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            local=False,
        )

    def stream(
        self, messages: list[dict], tools: "list[dict] | None" = None
    ) -> Iterator["str | Completion"]:
        # A print-mode CLI produces output only on exit, so this "stream"
        # is a single delta — correct for clients, just not incremental.
        completion = self.complete(messages)
        yield completion.text
        yield completion
