# Kultivait — Getting Started (for coding agents)

You are reading this because a human pasted a starter prompt into your chat.
Your job: guide them through installing kultivait and routing their first
prompt. Run commands with their permission, explain what each one does in a
line or two, and stop if anything fails.

## What kultivait is (tell them this first)

Kultivait is a local-first LLM routing proxy. Every prompt is weighed by a
local embedding model (`nomic-embed-text`, 274 MB, milliseconds) and routed to
the cheapest model that can carry it — their own local models first, the cloud
only when it earns its cost. Every decision is tallied in a savings ledger.
The greenest token is the one you never send.

## Step 1 — Install

```bash
curl -fsSL https://kultivait.ai/install.sh | sh
```

Or by hand:

```bash
uv tool install --from git+https://github.com/Standard-Pentest/kultivaite kultivait
```

## Step 2 — Initialize

```bash
kultivait init
```

This surveys THEIR machine: which local runtime is running (ollama, or
llama.cpp's `llama-server` in router mode), which models it has, and which
cloud CLIs (`claude` / `agy` / `gemini`) exist. The smallest capable model
becomes the simple tier; the largest becomes the reasoning tier. Decisions are
written to `~/.kultivait/config.toml` — editable by hand, and `init` can be
re-run anytime.

No cloud CLIs? That is fine — local-only mode is a first-class citizen (see
below).

## Step 3 — Serve

```bash
kultivait serve
```

Starts an OpenAI-compatible proxy at `http://localhost:4114`. Point one of
their tools at that endpoint — nothing else about the tool changes.

## Step 4 — Verify routing

```bash
kultivait route "why does this test deadlock?"
```

Dry-runs a classification: shows which tier that prompt would be routed to,
without sending anything.

## Step 5 — Watch the garden grow

```bash
kultivait harvest
```

Shows cumulative savings from `~/.kultivait/ledger.jsonl`, computed against
frontier-model baseline pricing. Add `--json` for raw data.

## If they have no cloud CLIs

Local-only mode is fully supported: cloud-worthy prompts are still recognized,
served by their best local model, and archived. Later, this:

```bash
kultivait escalations --brief
```

hands them a distilled, paste-ready brief to take to any frontier model
themselves.

## Bonus — phase-gate pruning

```bash
kultivait prune --from explore --to plan transcript.txt
```

Distills a transcript into a FINDINGS / DECISIONS / CONSTRAINTS / OPEN
QUESTIONS brief using a local model. The full transcript is always composted
to `~/.kultivait/compost/` — distillation is lossy, and the compost pile is
the escape hatch.

That's it. Get them to their first routed prompt, then show them the harvest.
