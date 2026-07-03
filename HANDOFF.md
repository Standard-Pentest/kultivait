# Kultivait — Session Handoff

**Date:** 2026-07-03
**For:** the next Fable-model session
**Repo:** github.com/Standard-Pentest/kultivaite · **Domain:** kultivait.ai

This is a self-contained handoff. You should be able to act on it without the
prior conversation. Project memory (deeper history) lives at
`~/.claude/projects/.../memory/kultivait-product-vision.md`.

---

## What Kultivait is

A local-first LLM routing proxy. It intercepts prompts on an OpenAI-compatible
`/v1/chat/completions` (and Anthropic-compatible `/v1/messages`) endpoint,
weighs each prompt with a local embedding (`nomic-embed-text`), and routes to
the cheapest model that can carry it — local ollama tiers first
(`llama3.1:8b`, `qwen3:14b`), cloud only when warranted (`claude`, `gemini`
via the `agy` CLI). It also distills context at phase gates (`kultivait prune`)
using a local model, and tallies savings in a ledger (`kultivait harvest`).

**Ethos (ordered):** reduce → right-size → localize. Honest about the math;
"cultivation, not a purity test." Context hygiene is the differentiator;
routing is the feature.

**Hardware constraint:** MacBook Pro M4, 24 GB RAM. 14B-class models fit;
30B MoE is too tight.

**Run it:**
```bash
uv run kultivait serve            # proxy on :4114
pi --provider kultivait --model auto   # dogfood client (config in ~/.pi/agent/models.json)
uv run kultivait harvest          # savings tally
uv run pytest                     # 27 tests, all passing
```

---

## What shipped this session

### 1. Context-window truncation fix (DONE, verified)

**The bug:** ollama silently truncates input to its default context window
(~8192 tokens). Every dogfood ledger entry showed `tokens_in` capped at
exactly **8191** — meaning classification and generation were running on a
*clipped* prompt. Agent clients like Pi wrap prompts in a ~5–8K-token envelope
(system prompt + tool defs), so real prompts routinely blew past the cap. The
distiller was worse off: it processes long transcripts, so truncation silently
dropped the tail *before* distilling.

**The fix:** `OllamaBackend` now sets `options.num_ctx` (default **32768**,
override with env `KULTIVAIT_NUM_CTX`). Applied in three places:
- `src/kultivait/backends.py` — `OllamaBackend.__init__(num_ctx=32768)` +
  `_payload()` sets `options.num_ctx` (covers both `complete` and `stream`).
- `src/kultivait/cli.py` — `NUM_CTX` constant; `_distill_generate` payload and
  `build_backends()` both use it.
- Tests: `tests/test_backends.py` (2 new: sets num_ctx, generous default).

**Verified:** an enveloped prompt that used to report exactly 8191 input
tokens now reports 8822 and scales up (a 16K-token test kept processing the
full prompt rather than truncating — slow, not clipped).

### Earlier this session (already shipped, for context)
- Streaming (OpenAI SSE + Anthropic event sequence).
- Anthropic `/v1/messages` endpoint (content-block + `system` normalization).
- Tool-call passthrough: `tools` forwarded to ollama; `tool_calls` returned;
  OpenAI↔ollama tool translators. Pi's full agentic loop (write + bash) ran
  through the proxy, local, at $0.00.
- `tool_fallback`: a tools-bearing request that classifies to a *cloud* tier
  falls back to the most capable *local* tool-capable tier, because the cloud
  CLIs (`claude -p`, `agy -p`) can't return client-side tool calls.

---

## The open decision: escalation advisory feature

### The tension it addresses
Because agentic clients (Pi) **always** send tool definitions, and cloud tiers
run as print-mode CLIs that can't participate in a client-side tool loop, the
`tool_fallback` **silently downgrades** every cloud-worthy agentic request to a
local model. Confirmed live: the "draft a technical spec" prompt classified to
`claude` (verified: `kultivait route` returns `tier: claude`, margin 0.045) but
was served by `qwen3:14b`. Through Pi, the cloud tier is effectively
unreachable. **User has decided AGAINST adding an Anthropic-API cloud backend
"at this time,"** so this stands as an accepted tradeoff — but the silent part
is the problem.

### User's proposed feature
Instead of silently downgrading, return a message advising the user to take
the prompt to a heavier tool (Claude / Antigravity) and naming the model to
choose. Kultivait as a "router of humans" for work beyond local capacity.

### My assessment (recommendation to carry forward)

**Verdict: build it — but NOT as an inline response-body message, and start
passive.**

**Why the instinct is right:**
- Strictly more honest than today's silent downgrade (which is the exact
  "dangerous misroute" the product tries to avoid).
- On-brand with the "reduce" principle: prevents the wasteful loop of
  ask-local → mediocre answer → re-ask → escalate anyway (3 generations, 1
  result).
- Cheap; needs no cloud API backend (respects the user's decision).

**The serious problem: it breaks the agent loop.** Inside Pi, a turn expects
`content` or `tool_calls` to continue — prose like "go use Claude" is treated
as the answer and derails the loop. The advice reaches a human only in
interactive chat, not mid-agent-turn, which is exactly when it would fire most.
Secondary risks: interruption tax on a "nothing changes" proxy; thin-margin
misclassifications (0.045!) interrupting for no real gain; hard-coded model
recommendations ("use Opus") go stale — the freshness problem.

**Recommended shape — same idea, out of band:**
1. **Passive first (build this next):** log every would-escalate decision to
   the ledger; surface in `kultivait harvest` — *"N prompts this session would
   likely have benefited from a frontier model: [list]."* Zero interruption,
   fully honest, and it reveals **how often this fires** before committing to
   any interrupt.
2. **Then, if wanted live:** a `kultivait.escalation` metadata field (human
   clients render it; agents ignore it) + optional desktop notification. Never
   the response body.
3. **The version worth getting excited about:** couple it with the pruning
   gate — instead of "go away," emit a *distilled, cloud-ready brief* of the
   context: "this is beyond local capacity — paste THIS into Claude." Routing +
   context hygiene compound into one feature: Kultivait knows its limits AND
   hands off cleanly/cheaply.

**Concrete next step:** implement the passive escalation log (TDD). Detect the
would-escalate condition (classification to a cloud tier, or `tool_fallback`
firing), record it to the ledger, and add an `escalations` section to
`harvest()` output. Small, reversible, and it generates the data to decide
whether the louder versions are worth it.

### STATUS UPDATE (2026-07-03, follow-up session)

**Triage of the "off the rails" note — RESOLVED, root cause confirmed.**
ollama server logs show the failing brainstorming-skill turn was truncated
(00:26:35: prompt=13223 → 8191, keep=4): ollama kept the first 4 tokens and
the most recent 8187, amputating ~5K tokens from the HEAD — where Pi's system
prompt and the skill instructions live. The model never saw the skill it was
running. Replaying the same conversation post-num_ctx-fix: the model now
answers "create it" with a correct contextual `write` tool call. Proxy
tool-call plumbing exonerated. Residual: one 36,413-token prompt clipped even
at 32K — truncation still possible, hence detection below.

**Passive escalation log + truncation detection — SHIPPED (31 tests).**
- Every ledger entry now records: `requested_tier`, `margin`, `escalated`,
  `tool_fallback`, `truncated`, `snippet` (first 80 chars of user text).
- `Completion.truncated` set by `is_truncated(prompt_eval_count, num_ctx)`
  (ollama pins prompt_eval_count at num_ctx - 1 when clipping).
- `harvest()` gains `escalations` {count, recent[]} and `truncated_inputs`.
- Live-verified with the user's actual spec prompt: requested=claude,
  served=qwen3:14b, tool_fallback=true, margin=0.0057 (razor-thin!).

**Next: the cloud-ready brief (option 3 above)** — on would-escalate, run the
pruning gate and emit a paste-ready distilled brief + recommended target,
surfaced via `kultivait escalations` / metadata, never the response body.

---

## Roadmap (unstarted, roughly ordered)
- Passive escalation log (see above) — recommended next.
- Route on user *intent*, not the agent envelope (classifier currently embeds
  the last user message, which works, but agent envelopes still shift things).
- Anthropic-endpoint tool support (`/v1/messages` tool_use — not yet done).
- Ambient gates via Claude Code hooks (PreCompact seam) — the "week 2" item.
- `install.sh`, Wh estimation in the ledger, learned centroids from real usage.

## Housekeeping notes
- Uncommitted at handoff: LICENSE (MIT), landing page `.dev`→`.ai`,
  `vercel.json`, this session's `num_ctx` fix. Review + commit when ready.
- Landing page deploys via Vercel (`vercel.json` rewrites `/` →
  `/landing/index.html`); user pending team access.
- Dogfood data is the current riskiest gap — real ledger entries > synthetic
  evals. Every misroute noticed is a labeled example for learned centroids.
