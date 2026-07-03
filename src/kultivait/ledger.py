"""Append-only JSONL ledger of routing decisions and the savings they earned."""

import json
import time
from pathlib import Path


class Ledger:
    def __init__(self, path: Path, baseline_in: float = 3.0, baseline_out: float = 15.0):
        self._path = Path(path)
        self._baseline_in = baseline_in  # USD per million input tokens at a frontier model
        self._baseline_out = baseline_out  # USD per million output tokens

    def record(
        self, *, tier: str, local: bool, tokens_in: int, tokens_out: int, cost_usd: float, **extra
    ) -> None:
        """Extra keyword fields (routing decision metadata, truncation flags,
        prompt snippets) are stored verbatim — the ledger is the analysis
        substrate, so silent failure modes must leave a trace here."""
        entry = {
            "ts": time.time(),
            "tier": tier,
            "local": local,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost_usd,
            **extra,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def harvest(self) -> dict:
        entries = []
        if self._path.exists():
            with self._path.open() as f:
                entries = [json.loads(line) for line in f if line.strip()]
        tokens_in = sum(e["tokens_in"] for e in entries)
        tokens_out = sum(e["tokens_out"] for e in entries)
        spent = sum(e["cost_usd"] for e in entries)
        baseline = (tokens_in * self._baseline_in + tokens_out * self._baseline_out) / 1e6
        escalations = [e for e in entries if e.get("tool_fallback")]
        return {
            "prompts": len(entries),
            "local_prompts": sum(1 for e in entries if e["local"]),
            "tokens_local": sum(e["tokens_in"] + e["tokens_out"] for e in entries if e["local"]),
            "spent_usd": spent,
            "baseline_usd": baseline,
            "saved_usd": baseline - spent,
            "escalations": {
                "count": len(escalations),
                "recent": [
                    {
                        "requested": e.get("requested_tier"),
                        "served": e["tier"],
                        "snippet": e.get("snippet", ""),
                    }
                    for e in escalations[-5:]
                ],
            },
            "truncated_inputs": sum(1 for e in entries if e.get("truncated")),
        }
