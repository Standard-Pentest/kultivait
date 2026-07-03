# kultivait

An intelligent LLM routing layer. Every prompt is weighed by a local embedding
model, routed to the cheapest model that can carry it — your own garden first,
the cloud only when it earns its cost — and tallied in a savings ledger.

**The greenest token is the one you never send.**

## How it works

1. **Intercept** — your tools point at kultivait's OpenAI-compatible endpoint.
2. **Weigh** — `nomic-embed-text` (274 MB, local, milliseconds) embeds the prompt
   and classifies it by cosine similarity to seed-prompt centroids. No cloud
   call decides whether to make a cloud call.
3. **Route** — trivial work → `llama3.1:8b`, local reasoning → `qwen3:14b`,
   doc-grounded checks → Gemini via `agy`, cross-file architecture → `claude`.
   Thin classification margins escalate one tier up: over-provisioning wastes
   cents, under-provisioning wastes an afternoon.
4. **Harvest** — every decision is recorded to `~/.kultivait/ledger.jsonl` with
   savings computed against frontier-model baseline pricing.

The routing approach was validated first: `experiments/routing_trust.py`
classified 24/24 held-out prompts correctly with zero dangerous misroutes
(cloud-worthy work sent to a weaker model).

## Usage

```bash
uv run kultivait serve --port 4114   # start the proxy
uv run kultivait route "why does this test deadlock?"   # dry-run a classification
uv run kultivait prune --from explore --to plan transcript.txt   # phase-gate handoff brief
uv run kultivait harvest             # cumulative savings
```

`prune` distills a transcript into a FINDINGS / DECISIONS / CONSTRAINTS /
OPEN QUESTIONS brief using a local model, so hygiene itself costs nothing.
The full transcript is always composted to `~/.kultivait/compost/` —
distillation is lossy, and the compost pile is the escape hatch. The same
operation is available on the proxy as `POST /gate`.

The distiller model was chosen by a planted-fact recall eval
(`experiments/distill_eval/`, 5 models x 2 prompts x 3 transcripts):

| model | mean recall | tokens kept | avg time |
|---|---|---|---|
| **gemma4:latest** (default) | **100%** | 61% | 29s |
| qwen3:14b | 96.3% | 44% | 15s |
| phi4:14b | 92.6% | 65% | 18s |
| qwen2.5:14b | 90.2% | 49% | 16s |
| llama3.1:8b | 86.5% | 44% | 8s |

Recall beats speed at a phase gate: a dropped constraint is catastrophic,
a slow gate is a coffee sip. Override with `KULTIVAIT_DISTILL_MODEL=qwen3:14b`
if you prefer the faster, tighter-compressing runner-up. A hardened
"never omit numbers" prompt variant was also tested and rejected — it traded
compression away for no recall gain; model choice dominated.

Point any OpenAI-compatible client at `http://localhost:4114/v1` with
`model: auto`. Both endpoints support streaming (SSE).

An Anthropic-compatible `/v1/messages` endpoint (streaming and
non-streaming, content blocks and `system` param handled) is also served,
so Anthropic-API clients can be pointed at the proxy:

```bash
ANTHROPIC_BASE_URL=http://localhost:4114 <your-tool>
```

Note: cloud tiers run through print-mode CLIs, which produce output only on
exit — those responses stream as a single final chunk. Local tiers stream
token-by-token.

### Using with the Pi coding agent

Add a provider to `~/.pi/agent/models.json`:

```json
"kultivait": {
  "api": "openai-completions",
  "apiKey": "kultivait",
  "baseUrl": "http://127.0.0.1:4114/v1",
  "models": [{ "contextWindow": 131072, "id": "auto", "input": ["text"] }]
}
```

Then: `pi --provider kultivait --model auto`. Tool calls pass through on the
OpenAI endpoint — Pi's full agentic loop (read/bash/edit/write) runs through
the proxy, with every turn routed and tallied.

Tool-bearing requests are always served by a local tool-capable tier, even
when classification points at a cloud tier: cloud CLIs run their own agent
loops and can't return client-side tool calls. The response's `kultivait`
metadata reports `tool_fallback: true` when this happens. Anthropic-endpoint
tool support is not yet implemented.

### Escalations: when the garden isn't enough

Every tool-fallback is also archived as an *escalation* — the full
conversation, saved instantly off the request path. When you decide a local
answer wasn't good enough:

```bash
kultivait escalations              # list cloud-worthy prompts served locally
kultivait escalations --brief      # distill the latest into a paste-ready brief
```

The brief (TASK / CONTEXT / PROGRESS / NEEDED) is distilled by your local
model and names the recommended target — "take this to Claude" — so
escalating costs one paste instead of re-explaining the whole session.
Routing knows its limits; hygiene makes the handoff cheap.

## Requirements

- [ollama](https://ollama.com) running locally with `nomic-embed-text`,
  `llama3.1:8b`, and `qwen3:14b` pulled
- `claude` and `agy` CLIs on PATH for the cloud tiers

## Development

```bash
uv run pytest
```

The landing page lives in `landing/index.html`.

## Roadmap

- Distillation-quality eval harness: automated planted-fact recall scoring
  across transcripts, to measure and improve the ~85% retention rate
- Ambient gates via agent-framework hooks (e.g. Claude Code hooks), so
  pruning happens at phase boundaries without manual invocation
- Streaming responses
- Watt-hour estimation in the ledger
- Learned centroids from your own routing history
