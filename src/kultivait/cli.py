"""kultivait CLI: serve the proxy, inspect the harvest, dry-run a route."""

import argparse
import json
import os
import sys
from pathlib import Path

import httpx
import numpy as np

from kultivait.backends import CLIBackend, OllamaBackend
from kultivait.escalations import (
    HANDOFF_PROMPT,
    RECOMMENDED_TARGETS,
    EscalationStore,
    render_transcript,
)
from kultivait.gates import Gate
from kultivait.ledger import Ledger
from kultivait.router import Router
from kultivait.seeds import CAPABILITY_ORDER, TIER_SEEDS

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
# gemma4 won the distillation eval: 100% planted-fact recall on all corpus
# transcripts vs 96.3% (qwen3:14b), 92.6% (phi4:14b best), 90.2% (qwen2.5:14b).
# See experiments/distill_eval/. Recall beats speed at a phase gate: a dropped
# constraint is catastrophic, a slow gate is a coffee sip.
DISTILL_MODEL = os.environ.get("KULTIVAIT_DISTILL_MODEL", "gemma4:latest")
# ollama truncates to a small default context; raise it so agent envelopes
# and long transcripts aren't silently clipped. Fits a 14B model on 24GB RAM.
NUM_CTX = int(os.environ.get("KULTIVAIT_NUM_CTX", "32768"))
LEDGER_PATH = Path.home() / ".kultivait" / "ledger.jsonl"
COMPOST_DIR = Path.home() / ".kultivait" / "compost"
ESCALATIONS_DIR = Path.home() / ".kultivait" / "escalations"


def _embed_batch(texts: list[str]) -> np.ndarray:
    r = httpx.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
    )
    r.raise_for_status()
    return np.array(r.json()["embeddings"])


def embed_one(text: str) -> np.ndarray:
    return _embed_batch([text])[0]


def build_router() -> Router:
    centroids = {}
    for tier, prompts in TIER_SEEDS.items():
        vecs = _embed_batch(prompts)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        centroids[tier] = vecs.mean(axis=0)
    return Router(centroids=centroids, capability_order=CAPABILITY_ORDER)


def _distill_generate(prompt: str) -> str:
    import re

    payload = {
        "model": DISTILL_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_ctx": NUM_CTX},
    }
    if DISTILL_MODEL.startswith("qwen3"):
        payload["think"] = False
    r = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=600)
    r.raise_for_status()
    text = r.json()["message"]["content"]
    # qwen3 may emit reasoning tags even with think disabled on older ollama
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def build_gate() -> Gate:
    return Gate(generate=_distill_generate, compost_dir=COMPOST_DIR)


def build_backends() -> dict:
    return {
        "llama3.1:8b": OllamaBackend("llama3.1:8b", OLLAMA_URL, num_ctx=NUM_CTX),
        "qwen3:14b": OllamaBackend("qwen3:14b", OLLAMA_URL, num_ctx=NUM_CTX),
        "claude": CLIBackend(["claude"], price_in=3.0, price_out=15.0),
        "gemini:agy": CLIBackend(["agy"], price_in=1.25, price_out=10.0),
    }


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from kultivait.server import create_app

    print("cultivating centroids from seed prompts...", file=sys.stderr)
    app = create_app(
        router=build_router(),
        embed=embed_one,
        backends=build_backends(),
        ledger=Ledger(LEDGER_PATH),
        gate=build_gate(),
        escalations=EscalationStore(ESCALATIONS_DIR),
    )
    print(f"kultivait listening on http://localhost:{args.port}", file=sys.stderr)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


def cmd_route(args: argparse.Namespace) -> None:
    router = build_router()
    decision = router.classify(embed_one(args.prompt))
    print(json.dumps(decision.__dict__, indent=2))


def cmd_prune(args: argparse.Namespace) -> None:
    transcript = Path(args.file).read_text() if args.file else sys.stdin.read()
    result = build_gate().distill(
        transcript, from_phase=args.from_phase, to_phase=args.to_phase
    )
    print(result.brief)
    print(
        f"\n--- pruned {result.tokens_before} -> {result.tokens_after} tokens "
        f"({100 * (1 - result.tokens_after / result.tokens_before):.0f}% composted, "
        f"recoverable: {result.compost_id})",
        file=sys.stderr,
    )


def cmd_escalations(args: argparse.Namespace) -> None:
    import datetime

    store = EscalationStore(ESCALATIONS_DIR)
    listed = store.list()
    if not listed:
        print("no escalations recorded — the local garden has been enough")
        return

    if not args.brief:
        for e in listed:
            when = datetime.datetime.fromtimestamp(e.ts).strftime("%m-%d %H:%M")
            print(f"{e.id}  {when}  wanted {e.requested_tier:<12}  {e.snippet}")
        print(f"\n{len(listed)} escalation(s). Distill one: kultivait escalations --brief [ID]")
        return

    target = args.id or listed[-1].id
    record = next(e for e in listed if e.id == target)
    transcript = render_transcript(store.load_messages(target))
    print(f"distilling {target} with {DISTILL_MODEL}...", file=sys.stderr)
    gate = Gate(generate=_distill_generate, compost_dir=COMPOST_DIR, template=HANDOFF_PROMPT)
    result = gate.distill(transcript, from_phase="local", to_phase="cloud")
    recommended = RECOMMENDED_TARGETS.get(record.requested_tier, record.requested_tier)
    print(f"# Escalation brief — take this to {recommended}\n")
    print(result.brief)
    print(
        f"\n--- {result.tokens_before} -> {result.tokens_after} tokens · "
        f"full conversation recoverable: {target}",
        file=sys.stderr,
    )


def cmd_harvest(args: argparse.Namespace) -> None:
    stats = Ledger(LEDGER_PATH).harvest()
    print(json.dumps(stats, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="kultivait")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the routing proxy")
    serve.add_argument("--port", type=int, default=4114)
    serve.set_defaults(func=cmd_serve)

    route = sub.add_parser("route", help="classify a prompt without executing it")
    route.add_argument("prompt")
    route.set_defaults(func=cmd_route)

    prune = sub.add_parser("prune", help="distill a transcript into a handoff brief")
    prune.add_argument("file", nargs="?", help="transcript file (default: stdin)")
    prune.add_argument("--from", dest="from_phase", default="previous")
    prune.add_argument("--to", dest="to_phase", default="next")
    prune.set_defaults(func=cmd_prune)

    esc = sub.add_parser(
        "escalations", help="list cloud-worthy prompts served locally; distill a handoff brief"
    )
    esc.add_argument("id", nargs="?", help="escalation id (default: most recent)")
    esc.add_argument("--brief", action="store_true", help="distill a paste-ready brief")
    esc.set_defaults(func=cmd_escalations)

    harvest = sub.add_parser("harvest", help="show cumulative savings")
    harvest.set_defaults(func=cmd_harvest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
