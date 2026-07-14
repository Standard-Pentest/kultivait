"""kultivait CLI: serve the proxy, inspect the harvest, dry-run a route.

Configuration is detected live from the machine (installed local models —
ollama or llama.cpp — and available CLIs) unless ~/.kultivait/config.toml
exists — `kultivait init` writes that file so the decisions are visible and
editable.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import httpx
import numpy as np

from kultivait.backends import CLIBackend, LlamaCppBackend, OllamaBackend
from kultivait.config import (
    KNOWN_CLIS,
    RUNTIME_URLS,
    Config,
    detect,
    load_config,
    save_config,
)
from kultivait.escalations import (
    HANDOFF_PROMPT,
    EscalationStore,
    recommended_target,
    render_transcript,
)
from kultivait.gates import Gate
from kultivait.ledger import Ledger
from kultivait.router import Router
from kultivait.seeds import ROLE_SEEDS

OLLAMA_URL = RUNTIME_URLS["ollama"]
LLAMACPP_URL = os.environ.get("KULTIVAIT_LLAMACPP_URL", RUNTIME_URLS["llamacpp"])
KULTIVAIT_HOME = Path.home() / ".kultivait"
CONFIG_PATH = KULTIVAIT_HOME / "config.toml"
LEDGER_PATH = KULTIVAIT_HOME / "ledger.jsonl"
COMPOST_DIR = KULTIVAIT_HOME / "compost"
ESCALATIONS_DIR = KULTIVAIT_HOME / "escalations"


def _survey_ollama() -> "tuple[list[str], dict[str, int]]":
    r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    r.raise_for_status()
    models = r.json().get("models", [])
    return [m["name"] for m in models], {m["name"]: m.get("size", 0) for m in models}


def match_gguf_sizes(names: "list[str]", files: "dict[str, int]") -> "dict[str, int]":
    """Router model ids and cache filenames drift (path prefixes flattened to
    underscores, :quant suffixes, case varies): match when every token of the
    id appears in the filename. Unmatched names are omitted — _param_billions
    still reads a parameter count from the name itself."""
    import re

    def tokens(s: str) -> set:
        return set(re.split(r"[/:_.\-]+", s.lower())) - {""}

    sizes: dict[str, int] = {}
    for name in names:
        n = tokens(name)
        for fname, size in files.items():
            if n <= tokens(fname):
                sizes[name] = size
                break
    return sizes


def _local_llamacpp_models(
    entries: "list[dict]", cache_files: "dict[str, int]"
) -> "tuple[list[str], dict[str, int]]":
    """Filter a router /v1/models listing to models actually on disk.

    Router listings include downloadable HF suggestions; picking a tier that
    isn't downloaded would trigger a surprise multi-GB fetch on first route.
    On-disk models carry `--model <path>` in status.args (stat that path);
    --hf-repo entries count only if a matching GGUF is already cached.
    """
    names: list[str] = []
    sizes: dict[str, int] = {}
    for m in entries:
        args = m.get("status", {}).get("args", [])
        path = Path(args[args.index("--model") + 1]) if "--model" in args else None
        if path and path.exists():
            names.append(m["id"])
            sizes[m["id"]] = path.stat().st_size
        elif "--hf-repo" in args:
            matched = match_gguf_sizes([m["id"]], cache_files)
            if m["id"] in matched:
                names.append(m["id"])
                sizes[m["id"]] = matched[m["id"]]
    return names, sizes


def _gguf_dirs() -> "list[Path]":
    """Where llama-server caches GGUF files, most specific override first."""
    override = os.environ.get("KULTIVAIT_LLAMACPP_MODELS_DIR") or os.environ.get(
        "LLAMA_CACHE"
    )
    if override:
        return [Path(override)]
    return [
        Path.home() / "Library" / "Caches" / "llama.cpp",  # macOS default
        Path.home() / ".cache" / "llama.cpp",
    ]


def _survey_llamacpp() -> "tuple[list[str], dict[str, int]]":
    """Names from the router's /v1/models (the authoritative request ids);
    sizes by stat-ing GGUF files on disk, because /v1/models reports full
    metadata only for currently-loaded models."""
    r = httpx.get(f"{LLAMACPP_URL}/v1/models", timeout=10)
    r.raise_for_status()
    entries = r.json().get("data", [])
    files: dict[str, int] = {}
    for d in _gguf_dirs():
        if d.expanduser().is_dir():
            for f in d.expanduser().rglob("*.gguf"):
                files[f.name] = f.stat().st_size
    return _local_llamacpp_models(entries, files)


def _reachable(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def _detect_runtime() -> str:
    """Prefer whichever local server is actually running; if both, ollama
    (the eval-proven setup). KULTIVAIT_RUNTIME overrides."""
    env = os.environ.get("KULTIVAIT_RUNTIME")
    if env:
        return env
    if _reachable(f"{OLLAMA_URL}/api/tags"):
        return "ollama"
    if _reachable(f"{LLAMACPP_URL}/v1/models"):
        return "llamacpp"
    return "ollama"


def _survey_local(runtime: str) -> "tuple[list[str], dict[str, int]]":
    return _survey_llamacpp() if runtime == "llamacpp" else _survey_ollama()


def _available_clis() -> "list[str]":
    return [cli for cli in KNOWN_CLIS if shutil.which(cli)]


def get_config() -> Config:
    if CONFIG_PATH.exists():
        config = load_config(CONFIG_PATH)
    else:
        runtime = _detect_runtime()
        models, sizes = _survey_local(runtime)
        config = detect(models, _available_clis(), sizes=sizes, runtime=runtime)
    # env overrides win, always
    distill = os.environ.get("KULTIVAIT_DISTILL_MODEL")
    num_ctx = os.environ.get("KULTIVAIT_NUM_CTX")
    if distill or num_ctx:
        from dataclasses import replace

        config = replace(
            config,
            distill_model=distill or config.distill_model,
            num_ctx=int(num_ctx) if num_ctx else config.num_ctx,
        )
    return config


def _require_embed_model(config: Config) -> str:
    if not config.embed_model:
        if config.runtime == "llamacpp":
            hint = (
                "Download a nomic-embed-text GGUF into your llama.cpp models dir\n"
                "and mark it `embedding = 1` in a --models-preset INI\n"
                "(see README: Using with llama.cpp), then retry."
            )
        else:
            hint = "Pull one (274 MB), then retry:\n\n    ollama pull nomic-embed-text"
        sys.exit(f"kultivait needs a local embedding model to weigh prompts.\n{hint}\n")
    return config.embed_model


def _embed_batch(config: Config, texts: "list[str]") -> np.ndarray:
    if config.runtime == "llamacpp":
        r = httpx.post(
            f"{config.embed_url()}/v1/embeddings",
            json={"model": config.embed_model, "input": texts},
            timeout=120,
        )
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        return np.array([d["embedding"] for d in data])
    r = httpx.post(
        f"{config.embed_url()}/api/embed",
        json={"model": config.embed_model, "input": texts},
        timeout=120,
    )
    r.raise_for_status()
    return np.array(r.json()["embeddings"])


def build_router(config: Config) -> Router:
    _require_embed_model(config)
    centroids = {}
    for tier in config.tiers:
        vecs = _embed_batch(config, ROLE_SEEDS[tier.role])
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        centroids[tier.name] = vecs.mean(axis=0)
    return Router(centroids=centroids, capability_order=config.capability_order())


def build_backends(config: Config) -> dict:
    backends = {}
    for tier in config.tiers:
        if tier.kind == "ollama":
            backends[tier.name] = OllamaBackend(
                tier.model, config.chat_base_url, num_ctx=config.num_ctx
            )
        elif tier.kind == "llamacpp":
            backends[tier.name] = LlamaCppBackend(tier.model, config.chat_base_url)
        elif tier.kind == "cli":
            backends[tier.name] = CLIBackend(
                tier.command, price_in=tier.price_in, price_out=tier.price_out
            )
        # "virtual" tiers get no backend: classified, never served — the
        # escalation path fires instead.
    return backends


def _distill_generate_for(config: Config):
    import re

    def generate(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        if config.runtime == "llamacpp":
            payload = {"model": config.distill_model, "messages": messages, "stream": False}
            r = httpx.post(
                f"{config.chat_base_url}/v1/chat/completions", json=payload, timeout=600
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"] or ""
        else:
            payload = {
                "model": config.distill_model,
                "messages": messages,
                "stream": False,
                "options": {"num_ctx": config.num_ctx},
            }
            if (config.distill_model or "").startswith("qwen3"):
                payload["think"] = False
            r = httpx.post(f"{config.chat_base_url}/api/chat", json=payload, timeout=600)
            r.raise_for_status()
            text = r.json()["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    return generate


def build_gate(config: Config, template: "str | None" = None) -> Gate:
    kwargs = {"template": template} if template else {}
    return Gate(generate=_distill_generate_for(config), compost_dir=COMPOST_DIR, **kwargs)


def cmd_init(args: argparse.Namespace) -> None:
    runtime = _detect_runtime()
    models, sizes = _survey_local(runtime)
    clis = _available_clis()
    config = detect(models, clis, sizes=sizes, runtime=runtime)

    print("kultivait surveyed your garden:\n")
    print(f"  local runtime: {runtime} ({config.chat_base_url})")
    print(f"  local models:  {len(models)} found")
    print(f"  cloud CLIs:    {', '.join(clis) if clis else 'none — local-only mode'}\n")
    for tier in config.tiers:
        if tier.kind == "virtual":
            served = "no backend — escalation briefs instead"
        elif tier.kind == "cli":
            served = f"{' '.join(tier.command)} (cloud, billed)"
        else:
            served = f"{tier.model} (local, free)"
        print(f"  {tier.role:<10} -> {served}")
    embed_missing = (
        "MISSING — download a nomic-embed GGUF"
        if runtime == "llamacpp"
        else "MISSING — run: ollama pull nomic-embed-text"
    )
    print(f"\n  embedding: {config.embed_model or embed_missing}")
    print(f"  distiller: {config.distill_model or 'MISSING — pull any 8B+ model'}")

    save_config(config, CONFIG_PATH)
    print(f"\nwrote {CONFIG_PATH}")
    print("edit it anytime; start the proxy with: kultivait serve")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from kultivait.server import create_app

    config = get_config()
    print("cultivating centroids from seed prompts...", file=sys.stderr)
    app = create_app(
        router=build_router(config),
        embed=lambda text: _embed_batch(config, [text])[0],
        backends=build_backends(config),
        ledger=Ledger(LEDGER_PATH),
        gate=build_gate(config),
        escalations=EscalationStore(ESCALATIONS_DIR),
    )
    port = args.port or config.port
    print(f"kultivait listening on http://localhost:{port}", file=sys.stderr)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def cmd_route(args: argparse.Namespace) -> None:
    config = get_config()
    router = build_router(config)
    decision = router.classify(_embed_batch(config, [args.prompt])[0])
    print(json.dumps(decision.__dict__, indent=2))


def cmd_prune(args: argparse.Namespace) -> None:
    transcript = Path(args.file).read_text() if args.file else sys.stdin.read()
    result = build_gate(get_config()).distill(
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

    config = get_config()
    target = args.id or listed[-1].id
    record = next(e for e in listed if e.id == target)
    transcript = render_transcript(store.load_messages(target))
    print(f"distilling {target} with {config.distill_model}...", file=sys.stderr)
    gate = build_gate(config, template=HANDOFF_PROMPT)
    result = gate.distill(transcript, from_phase="local", to_phase="cloud")
    print(f"# Escalation brief — take this to {recommended_target(record.requested_tier)}\n")
    print(result.brief)
    print(
        f"\n--- {result.tokens_before} -> {result.tokens_after} tokens · "
        f"full conversation recoverable: {target}",
        file=sys.stderr,
    )


def format_harvest(stats: dict) -> str:
    if stats["prompts"] == 0:
        return (
            "the harvest — nothing planted yet\n"
            "  start the proxy (kultivait serve) and route some work through it."
        )
    local_pct = round(100 * stats["local_prompts"] / stats["prompts"])
    lines = [
        "the harvest — season to date",
        "",
        f"  prompts routed     {stats['prompts']}  ({local_pct}% local)",
        f"  local tokens       {stats['tokens_local']:,}",
        f"  spent              ${stats['spent_usd']:.2f}",
        f"  frontier baseline  ${stats['baseline_usd']:.2f}",
        f"  kept in pocket     ${stats['saved_usd']:.2f}",
    ]
    esc = stats.get("escalations", {"count": 0, "recent": []})
    if esc["count"]:
        lines += ["", f"  {esc['count']} cloud-worthy prompt(s) served locally:"]
        for e in esc["recent"]:
            lines.append(f"    wanted {e['requested']}, served {e['served']}: {e['snippet']}")
        lines.append("    distill a handoff: kultivait escalations --brief")
    if stats.get("truncated_inputs"):
        lines += ["", f"  ⚠ {stats['truncated_inputs']} input(s) hit the context ceiling (raise num_ctx?)"]
    return "\n".join(lines)


def cmd_harvest(args: argparse.Namespace) -> None:
    stats = Ledger(LEDGER_PATH).harvest()
    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(format_harvest(stats))


def main() -> None:
    parser = argparse.ArgumentParser(prog="kultivait")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="survey this machine and write config")
    init.set_defaults(func=cmd_init)

    serve = sub.add_parser("serve", help="run the routing proxy")
    serve.add_argument("--port", type=int, default=None)
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
    harvest.add_argument("--json", action="store_true", help="machine-readable output")
    harvest.set_defaults(func=cmd_harvest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
