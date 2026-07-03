"""Cloud-ready escalation briefs.

When routing wants a cloud tier but a tools-bearing request must stay local
(tool_fallback), the full conversation is archived here instantly — off the
request path. The paste-ready brief is distilled lazily, at the moment the
human decides to escalate, when a 30-second wait is welcome rather than an
interruption.
"""

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

# Name tiers, not model versions: "use Opus" goes stale, "take it to Claude"
# doesn't.
RECOMMENDED_TARGETS = {
    "claude": "Claude (Claude Code or claude.ai)",
    "gemini:agy": "Gemini (agy or gemini.google.com)",
}

HANDOFF_PROMPT = """\
You are preparing a handoff brief for a more capable AI model. The
conversation below reached the limits of a local model. Distill it into a
brief the stronger model can act on immediately, with no other context.
Use exactly these sections:

TASK: (what the user ultimately wants, one paragraph)
CONTEXT: (facts, file paths, constraints, and exact values the model needs — verbatim where precision matters)
PROGRESS: (what has been tried or answered so far, and where it fell short)
NEEDED: (precisely what the stronger model should produce)

Be complete but ruthless: omit process narration and dead ends, never omit a
constraint, a file path, or an exact value.

TRANSCRIPT:
{transcript}
"""


@dataclass(frozen=True)
class Escalation:
    id: str
    ts: float
    requested_tier: str
    snippet: str


def render_transcript(messages: list[dict]) -> str:
    """Flatten an OpenAI-format conversation to readable text for distillation."""
    lines = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content") or ""
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            lines.append(f"[{role}] called {fn.get('name', '?')}({fn.get('arguments', '')})")
        if content:
            lines.append(f"[{role}] {content}")
    return "\n".join(lines)


class EscalationStore:
    def __init__(self, directory: Path):
        self._dir = Path(directory)

    def save(self, messages: list[dict], *, requested_tier: str) -> str:
        eid = f"esc-{uuid.uuid4().hex[:8]}"
        user_text = next(
            (m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        record = {
            "id": eid,
            "ts": time.time(),
            "requested_tier": requested_tier,
            "snippet": user_text[:80],
            "messages": messages,
        }
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / f"{eid}.json").write_text(json.dumps(record))
        return eid

    def _records(self) -> list[dict]:
        records = [json.loads(p.read_text()) for p in self._dir.glob("esc-*.json")]
        return sorted(records, key=lambda r: r["ts"])

    def list(self) -> list[Escalation]:
        return [
            Escalation(id=r["id"], ts=r["ts"], requested_tier=r["requested_tier"], snippet=r["snippet"])
            for r in self._records()
        ]

    def load_messages(self, eid: str) -> "list[dict]":  # quoted: list() method shadows builtin
        return json.loads((self._dir / f"{eid}.json").read_text())["messages"]
