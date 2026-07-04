# Copy Prompt Button + start.md Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Copy the starter prompt" CTA to the landing page (hero + final CTA) that copies an onboarding prompt to the clipboard, plus the `landing/start.md` doc that prompt points at.

**Architecture:** `landing/index.html` is a single static HTML file (inline CSS + inline JS, no build step, no test harness). We add one CSS block, two identical button markups, and one new JS handler. `landing/start.md` is a standalone markdown file deployed next to `index.html` and `install.sh`.

**Tech Stack:** Plain HTML/CSS/JS. Verification via `grep` assertions and a final in-browser check.

**Spec:** `docs/superpowers/specs/2026-07-03-copy-prompt-button-design.md`

## Global Constraints

- The copied prompt text, verbatim (em dash, trailing period, `www.` host): `Read https://www.kultivait.ai/start.md and help me plant my first garden — routing my prompts to the cheapest model that can carry them.`
- Button label: `Copy the starter prompt`. Success label: `✓ Copied — now paste into your agent`. Failure label: `Press ⌘C — prompt selected`. Subtitle: `paste it into your coding agent for a guided walkthrough`.
- Use only existing CSS variables (`--copper`, `--copper-bright`, `--moss-deep`, `--cream-muted`, `--mono`). No new fonts, no new colors, no external assets.
- Do NOT touch the existing `[data-copy]` handler or the install-chip buttons.
- Every command in `start.md` must appear in `README.md` — no invented commands or flags.
- All work happens on branch `feat/copy-prompt-button` (created in Task 1).

---

### Task 1: Create `landing/start.md`

**Files:**
- Create: `landing/start.md`

**Interfaces:**
- Produces: `landing/start.md`, deployed at `https://kultivait.ai/start.md` — the target of the copied prompt. No code depends on it; Tasks 2–4 only reference its URL.

- [ ] **Step 1: Create the working branch**

```bash
git checkout -b feat/copy-prompt-button
```

- [ ] **Step 2: Write the file**

Create `landing/start.md` with exactly this content:

````markdown
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

This surveys THEIR machine: which ollama models are pulled, which cloud CLIs
(`claude` / `agy` / `gemini`) exist. The smallest capable model becomes the
simple tier; the largest becomes the reasoning tier. Decisions are written to
`~/.kultivait/config.toml` — editable by hand, and `init` can be re-run
anytime.

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
````

- [ ] **Step 3: Verify every command exists in README.md**

```bash
for cmd in "curl -fsSL https://kultivait.ai/install.sh | sh" \
           "uv tool install --from git+https://github.com/Standard-Pentest/kultivaite kultivait" \
           "kultivait init" "kultivait serve" "kultivait harvest" \
           "kultivait escalations" "kultivait prune --from explore --to plan"; do
  grep -qF "$cmd" README.md && echo "OK: $cmd" || echo "MISSING FROM README: $cmd"
done
```

Expected: seven `OK:` lines, zero `MISSING` lines. If anything is missing, fix `start.md` to match the README (the README is the source of truth), not the other way around.

- [ ] **Step 4: Verify length target**

```bash
wc -l landing/start.md
```

Expected: under 120 lines.

- [ ] **Step 5: Commit**

```bash
git add landing/start.md
git commit -m "feat(landing): add start.md agent-facing onboarding doc"
```

---

### Task 2: Button CSS and markup (hero + final CTA)

**Files:**
- Modify: `landing/index.html` (CSS block near line 166, hero markup near line 517, final-CTA markup near line 793)

**Interfaces:**
- Consumes: existing CSS variables and the `.install` block positions in `landing/index.html`.
- Produces: two `<button class="prompt-copy" data-copy-prompt="…">` elements, each containing `<span class="prompt-copy-label">`, and a sibling `<span class="prompt-copy-fallback">` holding the prompt text. Task 3's JS selects `[data-copy-prompt]`, `.prompt-copy-label`, and `.prompt-copy-fallback` — these class/attribute names are load-bearing.

- [ ] **Step 1: Add the CSS**

In `landing/index.html`, find this existing rule (around line 166):

```css
  .install button:hover { color: var(--linen); border-color: var(--copper); }
```

Insert immediately AFTER it:

```css
  .prompt-start { margin-top: 14px; max-width: 520px; position: relative; }
  .prompt-copy {
    display: inline-flex; align-items: center; gap: 10px;
    background: none; border: 1px solid var(--copper);
    color: var(--copper-bright); border-radius: 6px;
    font-family: var(--mono); font-size: 13.5px;
    padding: 11px 18px; cursor: pointer; transition: all 0.2s;
  }
  .prompt-copy:hover {
    color: var(--moss-deep); background: var(--copper-bright);
    border-color: var(--copper-bright);
  }
  .prompt-copy-sub {
    display: block; margin-top: 8px;
    font-family: var(--mono); font-size: 12px;
    color: var(--cream-muted); letter-spacing: 0.02em;
  }
  .prompt-copy-fallback {
    position: absolute; width: 1px; height: 1px;
    overflow: hidden; clip-path: inset(50%); white-space: nowrap;
  }
```

Notes: `.prompt-copy-fallback` is visually hidden but remains selectable (no `display:none`/`visibility:hidden`, which would break range selection). The button inherits the page's global `button:focus-visible` outline rule — do not add a custom focus style.

- [ ] **Step 2: Add the hero instance**

In the hero, find:

```html
        <div class="install" id="install">
          <span class="dollar" aria-hidden="true">$</span>
          <code>curl -fsSL https://kultivait.ai/install.sh | sh</code>
          <button type="button" data-copy="curl -fsSL https://kultivait.ai/install.sh | sh">copy</button>
        </div>
        <p class="hero-note">
```

Insert between the closing `</div>` of the install block and the `<p class="hero-note">`:

```html
        <div class="prompt-start">
          <button type="button" class="prompt-copy" data-copy-prompt="Read https://www.kultivait.ai/start.md and help me plant my first garden — routing my prompts to the cheapest model that can carry them.">
            <span aria-hidden="true">⧉</span>
            <span class="prompt-copy-label" aria-live="polite">Copy the starter prompt</span>
          </button>
          <span class="prompt-copy-sub">paste it into your coding agent for a guided walkthrough</span>
          <span class="prompt-copy-fallback" tabindex="-1">Read https://www.kultivait.ai/start.md and help me plant my first garden — routing my prompts to the cheapest model that can carry them.</span>
        </div>
```

- [ ] **Step 3: Add the final-CTA instance (with grid wrapper)**

`.final-cta` is a two-column grid (`grid-template-columns: 1fr auto`) whose children are the text `<div>` and the `.install` div. A third direct child would become a third grid cell and break the layout, so wrap the install block and the new button together. Find:

```html
        <div class="install">
          <span class="dollar" aria-hidden="true">$</span>
          <code>curl -fsSL https://kultivait.ai/install.sh | sh</code>
          <button type="button" data-copy="curl -fsSL https://kultivait.ai/install.sh | sh">copy</button>
        </div>
      </div>
    </div>
  </section>
</main>
```

Replace with:

```html
        <div>
          <div class="install">
            <span class="dollar" aria-hidden="true">$</span>
            <code>curl -fsSL https://kultivait.ai/install.sh | sh</code>
            <button type="button" data-copy="curl -fsSL https://kultivait.ai/install.sh | sh">copy</button>
          </div>
          <div class="prompt-start">
            <button type="button" class="prompt-copy" data-copy-prompt="Read https://www.kultivait.ai/start.md and help me plant my first garden — routing my prompts to the cheapest model that can carry them.">
              <span aria-hidden="true">⧉</span>
              <span class="prompt-copy-label" aria-live="polite">Copy the starter prompt</span>
            </button>
            <span class="prompt-copy-sub">paste it into your coding agent for a guided walkthrough</span>
            <span class="prompt-copy-fallback" tabindex="-1">Read https://www.kultivait.ai/start.md and help me plant my first garden — routing my prompts to the cheapest model that can carry them.</span>
          </div>
        </div>
      </div>
    </div>
  </section>
</main>
```

(The `.final-cta .install { margin-top: 0; }` rule still applies — the selector is descendant, not child.)

- [ ] **Step 4: Verify markup counts**

```bash
grep -c 'data-copy-prompt=' landing/index.html   # expected: 2
grep -c 'prompt-copy-fallback' landing/index.html # expected: 3 (1 CSS rule + 2 spans)
grep -c 'data-copy="curl'      landing/index.html # expected: 2 (unchanged)
grep -c 'Read https://www.kultivait.ai/start.md and help me plant my first garden — routing my prompts to the cheapest model that can carry them.' landing/index.html # expected: 4 (2 attrs + 2 fallback spans)
```

All four counts must match. The last check also proves all four copies of the prompt string are byte-identical.

- [ ] **Step 5: Commit**

```bash
git add landing/index.html
git commit -m "feat(landing): copy-prompt button markup and styles (hero + final CTA)"
```

---

### Task 3: Clipboard JS handler

**Files:**
- Modify: `landing/index.html` (the `<script>` block near the bottom, after the existing copy-buttons handler)

**Interfaces:**
- Consumes: the `[data-copy-prompt]` buttons, `.prompt-copy-label` spans, and sibling `.prompt-copy-fallback` spans from Task 2. Reads the prompt from `btn.dataset.copyPrompt` (the camelCase form of `data-copy-prompt`).
- Produces: nothing consumed by later tasks; Task 4 exercises this behavior in a browser.

- [ ] **Step 1: Add the handler**

In the `<script>` block at the bottom of `landing/index.html`, find:

```js
  // copy buttons
  document.querySelectorAll("[data-copy]").forEach(btn => {
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(btn.dataset.copy);
        btn.textContent = "copied";
        setTimeout(() => { btn.textContent = "copy"; }, 1600);
      } catch { btn.textContent = "select it"; }
    });
  });
```

Insert immediately AFTER it (do not modify the block above — `[data-copy]` does not match `data-copy-prompt`, so the two handlers never overlap):

```js
  // starter-prompt buttons — swap the label span only, keep icon intact
  document.querySelectorAll("[data-copy-prompt]").forEach(btn => {
    const label = btn.querySelector(".prompt-copy-label");
    const original = label.textContent;
    let timer;
    btn.addEventListener("click", async () => {
      clearTimeout(timer);
      try {
        await navigator.clipboard.writeText(btn.dataset.copyPrompt);
        label.textContent = "✓ Copied — now paste into your agent";
      } catch {
        const fallback = btn.parentElement.querySelector(".prompt-copy-fallback");
        const range = document.createRange();
        range.selectNodeContents(fallback);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        label.textContent = "Press ⌘C — prompt selected";
      }
      timer = setTimeout(() => { label.textContent = original; }, 1800);
    });
  });
```

- [ ] **Step 2: Verify with grep**

```bash
grep -c 'data-copy-prompt\]' landing/index.html          # expected: 1 (the selector)
grep -c 'dataset.copyPrompt' landing/index.html          # expected: 1
grep -c '✓ Copied — now paste into your agent' landing/index.html  # expected: 1
```

- [ ] **Step 3: Syntax-check the inline script**

```bash
python3 - <<'EOF'
import re, subprocess, tempfile, pathlib
html = pathlib.Path("landing/index.html").read_text()
js = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
p = tempfile.NamedTemporaryFile(suffix=".js", delete=False, mode="w"); p.write(js); p.close()
r = subprocess.run(["node", "--check", p.name])
print("JS SYNTAX OK" if r.returncode == 0 else "JS SYNTAX ERROR")
EOF
```

Expected: `JS SYNTAX OK`. (If `node` is unavailable, skip this step; Task 4's browser check covers it.)

- [ ] **Step 4: Commit**

```bash
git add landing/index.html
git commit -m "feat(landing): clipboard handler for copy-prompt buttons"
```

---

### Task 4: In-browser verification

**Files:**
- Modify: `landing/index.html` only if fixes are needed.

**Interfaces:**
- Consumes: everything from Tasks 2–3, served over HTTP (the clipboard API requires a secure context; `http://localhost` qualifies).

- [ ] **Step 1: Serve the page**

```bash
python3 -m http.server 8765 --directory landing &
```

Expected: server starts; `http://localhost:8765/` loads the page and `http://localhost:8765/start.md` returns the doc (not a 404).

- [ ] **Step 2: Verify click behavior in a browser**

Open `http://localhost:8765/` in a browser (use the available browser automation MCP — chrome-devtools or playwright — if present; otherwise do this manually and report results). For EACH of the two buttons:

1. Click it.
2. Read the clipboard (in DevTools console: `await navigator.clipboard.readText()` after a user gesture, or paste into an input). Expected, exactly: `Read https://www.kultivait.ai/start.md and help me plant my first garden — routing my prompts to the cheapest model that can carry them.`
3. Confirm the label reads `✓ Copied — now paste into your agent` and the `⧉` icon is still visible next to it.
4. Wait 2 seconds; confirm the label reverts to `Copy the starter prompt`.
5. Double-click rapidly; confirm the label doesn't get stuck on the copied state (the `clearTimeout` guard).

- [ ] **Step 3: Verify the clipboard-failure fallback**

In the DevTools console, force the failure path, then click the hero button:

```js
navigator.clipboard.writeText = () => Promise.reject(new Error("denied"));
```

Expected: the label reads `Press ⌘C — prompt selected`, and the full prompt text is selected on the page (check `window.getSelection().toString()` — it must equal the prompt). Reload the page afterward to restore the real clipboard API.

- [ ] **Step 4: Verify keyboard access**

Tab to the hero button — the copper `:focus-visible` outline must appear. Press Enter — the copy fires and the label swaps.

- [ ] **Step 5: Verify layout at narrow width**

Resize to ≤860 px (the `.final-cta` breakpoint). The final-CTA section must stack in one column with the button below the install line, nothing overflowing.

- [ ] **Step 6: Stop the server and finish**

```bash
kill %1
```

If steps 2–5 exposed problems, fix them in `landing/index.html`, re-run the failing check, and commit with message `fix(landing): <what was broken> in copy-prompt button`. If everything passed with no changes, no commit is needed for this task.
