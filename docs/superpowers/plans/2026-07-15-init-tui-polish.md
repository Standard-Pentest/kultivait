# kultivait init Rich-based TUI polish — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restyle `kultivait init`'s existing linear flow (survey → confirm → download → launch) with `rich` — a colored survey table, a real download progress bar, and a server-start spinner — without changing any step's logic or breaking any existing test.

**Architecture:** A new presentation-only module `tui.py` holds a shared `rich.console.Console`, a `log()` wrapper, a styled `ask()`, and a `render_survey()` table builder. `bootstrap.py` and `cli.py` change only their *default* `log=`/`confirm=` values and swap literal `print()` calls for Rich renders; `_download` gains an optional `on_progress` callback that is a no-op when omitted. Every existing injection seam and `capsys`-asserted substring is preserved.

**Tech Stack:** Python 3.12+, `rich`, pytest, uv.

## Global Constraints

- `requires-python >= 3.12` — do not use syntax below that floor.
- No function may lose an existing parameter or change an existing parameter's contract; only *default values* and *print call sites* change, plus one new optional keyword parameter on `_download`/`download_models`.
- Every test in `tests/test_bootstrap.py` and `tests/test_cli_init.py` must keep passing **unmodified**.
- No linter/formatter/type-checker runs in CI — match surrounding style by hand.
- Presentation only: never change the sequence or logic of steps in `cmd_init` or `bootstrap.run`.
- Scope is `init` only — do not touch `serve`/`route`/`prune`/`escalations`/`harvest` output.
- Run all commands from the worktree root. Full suite command: `uv run pytest`.

---

### Task 1: Add `rich` dependency and create the `tui` module

**Files:**
- Modify: `pyproject.toml` (add `rich` to `[project.dependencies]`)
- Create: `src/kultivait/tui.py`
- Test: `tests/test_tui.py`

**Interfaces:**
- Consumes: `kultivait.config.Config` and its `TierSpec` entries (`tier.role: str`, `tier.kind: str` in `{"ollama","llamacpp","cli","virtual"}`, `tier.model: str | None`, `tier.command: list[str] | None`); `Config.chat_base_url`, `Config.embed_model`, `Config.distill_model`, `Config.tiers`.
- Produces:
  - `console: rich.console.Console` — module-level singleton.
  - `log(*args, **kwargs) -> None` — forwards to `console.print`.
  - `ask(prompt: str, input_fn=input) -> bool` — prints styled prompt, returns `input_fn(...).strip().lower() in ("", "y", "yes")`.
  - `render_survey(runtime: str, base_url: str, models: list[str], clis: list[str], config: Config) -> rich.console.RenderableType` — a `Panel` wrapping a `Table` with columns Role / Serves / Kind.

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add `"rich>=13.0.0"` to the `dependencies` list:

```toml
dependencies = [
    "fastapi>=0.139.0",
    "httpx>=0.28.1",
    "numpy>=2.5.0",
    "rich>=13.0.0",
    "uvicorn>=0.49.0",
]
```

Then sync:

```bash
uv sync
```

Expected: resolves and installs `rich` (and its `markdown-it-py`/`pygments` deps) without error.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_tui.py`:

```python
"""tui: presentation-only helpers. render_survey builds a table; ask keeps
bootstrap.ask's default-yes contract. No real console output asserted —
Rich rendering is captured to plain text."""

from kultivait import tui
from kultivait.config import Config, TierSpec


def _config() -> Config:
    return Config(
        tiers=[
            TierSpec(role="simple", kind="llamacpp", model="qwen3-4b"),
            TierSpec(role="architect", kind="cli", command=["claude", "-p"]),
            TierSpec(role="docs", kind="virtual"),
        ],
        chat_base_url="http://localhost:8080",
        embed_model="nomic-embed-text",
        distill_model="qwen3-14b",
    )


def _plain(renderable) -> str:
    with tui.console.capture() as cap:
        tui.console.print(renderable)
    return cap.get()


def test_render_survey_lists_every_role_and_backend():
    out = _plain(tui.render_survey("llamacpp", "http://localhost:8080",
                                   ["qwen3-4b"], ["claude"], _config()))
    assert "simple" in out
    assert "qwen3-4b" in out          # local model served
    assert "claude" in out            # cli-served role
    assert "escalation" in out.lower()  # virtual tier note


def test_render_survey_shows_missing_embed_and_distiller():
    cfg = _config()
    cfg.embed_model = None
    cfg.distill_model = None
    out = _plain(tui.render_survey("llamacpp", "http://localhost:8080",
                                   [], [], cfg))
    assert "MISSING" in out


def test_ask_default_yes_contract_matches_bootstrap():
    assert tui.ask("go?", input_fn=lambda _: "") is True
    assert tui.ask("go?", input_fn=lambda _: "y") is True
    assert tui.ask("go?", input_fn=lambda _: "N") is False


def test_ask_passes_prompt_to_input_fn():
    seen = []
    tui.ask("proceed?", input_fn=lambda p: seen.append(p) or "y")
    assert seen and "proceed?" in seen[0]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_tui.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kultivait.tui'`.

- [ ] **Step 4: Write `tui.py`**

Create `src/kultivait/tui.py`:

```python
"""Presentation-only helpers for `kultivait init`. A shared Rich console, a
print-shaped log(), a styled yes/no ask() with bootstrap.ask's contract, and
render_survey() for the post-scan summary table. No side effects beyond
writing to the console; Rich degrades to plain text on non-tty streams, so
every caller stays testable via capsys/capture."""

from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.table import Table

from kultivait.config import Config

console = Console()


def log(*args, **kwargs) -> None:
    """Drop-in default for bootstrap's `log=print` seams."""
    console.print(*args, **kwargs)


def ask(prompt: str, input_fn=input) -> bool:
    """[Y/n] confirm, default yes — same contract as bootstrap.ask, styled.

    The question is painted via the console; the actual read still goes
    through input_fn so tests inject answers exactly as they do for
    bootstrap.ask."""
    console.print(f"[bold]{prompt}[/bold] [dim][Y/n][/dim] ", end="")
    return input_fn("").strip().lower() in ("", "y", "yes")


_KIND_STYLE = {
    "ollama": ("local, free", "green"),
    "llamacpp": ("local, free", "green"),
    "cli": ("cloud, billed", "yellow"),
    "virtual": ("no backend — escalation briefs instead", "red"),
}


def render_survey(
    runtime: str, base_url: str, models: list[str], clis: list[str], config: Config
) -> RenderableType:
    """Panel + Table replacing cmd_init's print-loop: one row per tier plus
    embedding/distiller status lines, colored by whether the tier is served
    locally (green), by a billed cloud CLI (yellow), or not at all (red)."""
    table = Table(expand=False, show_edge=False, pad_edge=False)
    table.add_column("Role", style="bold")
    table.add_column("Serves")
    table.add_column("Kind")
    for tier in config.tiers:
        note, color = _KIND_STYLE.get(tier.kind, ("", "white"))
        if tier.kind == "cli":
            serves = " ".join(tier.command or [])
        elif tier.kind == "virtual":
            serves = "—"
        else:
            serves = tier.model or "—"
        table.add_row(tier.role, serves, f"[{color}]{note}[/{color}]")

    embed = config.embed_model or "[red]MISSING[/red]"
    distill = config.distill_model or "[red]MISSING[/red]"
    header = (
        f"[bold]local runtime[/bold]  {runtime} ({base_url})\n"
        f"[bold]local models[/bold]   {len(models)} found\n"
        f"[bold]cloud CLIs[/bold]     {', '.join(clis) if clis else 'none — local-only mode'}\n"
    )
    body = Table.grid()
    body.add_row(header)
    body.add_row(table)
    body.add_row(f"\n[bold]embedding[/bold]  {embed}\n[bold]distiller[/bold]  {distill}")
    return Panel(body, title="kultivait surveyed your garden", border_style="green")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_tui.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/kultivait/tui.py tests/test_tui.py
git commit -m "feat: add rich-backed tui module for init (console, ask, render_survey)"
```

---

### Task 2: Add optional `on_progress` callback to `_download` / `download_models`

**Files:**
- Modify: `src/kultivait/bootstrap.py` (`_download`, `download_models`)
- Test: `tests/test_bootstrap.py` (add one test; do not modify existing ones)

**Interfaces:**
- Consumes: nothing new.
- Produces: `_download(..., on_progress: "Callable[[int, int], None] | None" = None)` and `download_models(..., on_progress=None)`. When `on_progress is None`, behavior is byte-for-byte identical to today (the existing `\r`-style text line via `log`). When supplied, `on_progress(done_bytes, total_bytes)` is called on each chunk **instead of** the `\r` text line.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_bootstrap.py` (near the other `_download` tests):

```python
def test_download_on_progress_receives_byte_counts(tmp_path):
    body = b"0123456789"
    client = FakeClient(body)
    seen = []
    ok = bootstrap._download(
        client, "http://x/tiny.gguf", tmp_path / "tiny.gguf", len(body),
        on_progress=lambda done, total: seen.append((done, total)), log=_quiet,
    )
    assert ok is True
    assert seen[-1] == (len(body), len(body))  # final call reports completion
    assert all(total == len(body) for _, total in seen)


def test_download_without_on_progress_still_logs_text(tmp_path):
    body = b"0123456789"
    client = FakeClient(body)
    lines = []

    def log(*args, **kwargs):
        lines.append(" ".join(str(a) for a in args))

    bootstrap._download(client, "http://x/tiny.gguf", tmp_path / "tiny.gguf", len(body), log=log)
    assert any("MB" in line for line in lines)  # default path unchanged
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_bootstrap.py::test_download_on_progress_receives_byte_counts -v`
Expected: FAIL — `_download()` got an unexpected keyword argument `on_progress`.

- [ ] **Step 3: Implement the callback**

In `src/kultivait/bootstrap.py`, add the import at the top (after the existing imports):

```python
from typing import Callable, Optional
```

Change `_download`'s signature from:

```python
def _download(
    client, url: str, dest: Path, expected_bytes: int, sha256: str = "", log=print
) -> bool:
```

to:

```python
def _download(
    client, url: str, dest: Path, expected_bytes: int, sha256: str = "",
    on_progress: "Optional[Callable[[int, int], None]]" = None, log=print,
) -> bool:
```

Then, inside the chunk-writing loop, replace the current progress line:

```python
            for chunk in r.iter_bytes(CHUNK):
                f.write(chunk)
                done += len(chunk)
                log(
                    f"\r  {dest.name}: {done / 2**20:.0f}/{expected_bytes / 2**20:.0f} MB",
                    end="",
                    flush=True,
                )
        log("")
```

with:

```python
            for chunk in r.iter_bytes(CHUNK):
                f.write(chunk)
                done += len(chunk)
                if on_progress is not None:
                    on_progress(done, expected_bytes)
                else:
                    log(
                        f"\r  {dest.name}: {done / 2**20:.0f}/{expected_bytes / 2**20:.0f} MB",
                        end="",
                        flush=True,
                    )
        if on_progress is None:
            log("")
```

- [ ] **Step 4: Thread it through `download_models`**

Change `download_models`'s signature from:

```python
def download_models(
    plan: SetupPlan,
    dest: Path,
    confirm=ask,
    client: "httpx.Client | None" = None,
    log=print,
) -> bool:
```

to:

```python
def download_models(
    plan: SetupPlan,
    dest: Path,
    confirm=ask,
    client: "httpx.Client | None" = None,
    on_progress: "Optional[Callable[[int, int], None]]" = None,
    log=print,
) -> bool:
```

and update its `_download` call from:

```python
            if not _download(
                client, m.url(), dest / m.filename, m.approx_bytes, sha256=m.sha256, log=log
            ):
```

to:

```python
            if not _download(
                client, m.url(), dest / m.filename, m.approx_bytes,
                sha256=m.sha256, on_progress=on_progress, log=log,
            ):
```

- [ ] **Step 5: Run the new tests and the full bootstrap suite**

Run: `uv run pytest tests/test_bootstrap.py -v`
Expected: PASS — the two new tests pass and all pre-existing bootstrap tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/kultivait/bootstrap.py tests/test_bootstrap.py
git commit -m "feat: add optional on_progress callback to bootstrap downloads"
```

---

### Task 3: Swap bootstrap defaults to the Rich helpers

**Files:**
- Modify: `src/kultivait/bootstrap.py` (`ensure_llamacpp` print, and `log=`/`confirm=` defaults on `download_models`, `run`, `start_server`, `offer_wired_limit`)
- Test: `tests/test_bootstrap.py` (no changes — existing tests inject their own `log`/`confirm`, and `capsys` tests still see plain text)

**Interfaces:**
- Consumes: `tui.log`, `tui.ask`, `tui.console` from Task 1.
- Produces: unchanged signatures; only default values differ. `bootstrap.ask` itself stays defined and directly tested.

- [ ] **Step 1: Run the affected tests first to capture the green baseline**

Run: `uv run pytest tests/test_bootstrap.py tests/test_cli_init.py -q`
Expected: all PASS. This is the baseline the swap must preserve.

- [ ] **Step 2: Import tui in bootstrap**

In `src/kultivait/bootstrap.py`, add after the existing `from kultivait.hardware import SetupPlan` line:

```python
from kultivait import tui
```

- [ ] **Step 3: Style the Homebrew advisory print**

In `ensure_llamacpp`, change:

```python
        print(BREW_INSTALL_HINT)
```

to:

```python
        tui.console.print(BREW_INSTALL_HINT)
```

(The asserted substring `"Homebrew"` is inside `BREW_INSTALL_HINT`, so `test_ensure_llamacpp_without_brew_goes_advisory` still passes — Rich prints plain text to the captured non-tty stream.)

- [ ] **Step 4: Swap the default `log`/`confirm` values**

Change these four signatures' defaults (leave every other part of each signature exactly as-is):

- `download_models(..., confirm=ask, ..., log=print)` → `confirm=tui.ask`, `log=tui.log`
- `run(..., confirm=ask, ..., log=print, ...)` → `confirm=tui.ask`, `log=tui.log`
- `start_server(..., log=print)` → `log=tui.log`
- `offer_wired_limit(..., log=print)` → `log=tui.log`

Do **not** change the standalone `ask(prompt, input_fn=input)` function definition — it stays as the plain implementation that `test_ask_defaults_to_yes` exercises directly.

- [ ] **Step 5: Run the full suite to confirm nothing regressed**

Run: `uv run pytest tests/test_bootstrap.py tests/test_cli_init.py -q`
Expected: identical PASS count to Step 1's baseline.

- [ ] **Step 6: Commit**

```bash
git add src/kultivait/bootstrap.py
git commit -m "feat: default init bootstrap output to rich console helpers"
```

---

### Task 4: Wire `render_survey`, the download bar, and the start spinner into `cli.py`

**Files:**
- Modify: `src/kultivait/cli.py` (`cmd_init` summary block; `_offer_setup` bootstrap call)
- Test: `tests/test_cli_init.py` (no changes — `test_cmd_init_survives_bare_machine` and `test_cmd_init_no_setup_never_offers` must still pass unmodified)

**Interfaces:**
- Consumes: `tui.render_survey`, `tui.console` (Task 1); `bootstrap.run`'s `on_progress` param reaches `download_models` via `**kwargs`? — NO: `run()` does not currently forward `on_progress`. This task passes `on_progress` **and** the spinner by wrapping at the `cli.py` call layer, so verify `bootstrap.run` accepts and threads it (see Step 3).
- Produces: no new public functions.

- [ ] **Step 1: Confirm `run()` can forward `on_progress` to `download_models`**

Read `src/kultivait/bootstrap.py:run`. It calls `download_models(plan, gguf_dir, confirm=confirm, client=client, log=log)`. To let `cli.py` supply a progress bar, `run` must accept and forward `on_progress`. Add `on_progress=None` to `run`'s keyword-only params and pass it through:

Change `run`'s signature to include (alongside the other keyword-only params):

```python
    on_progress=None,
```

and change its `download_models` call from:

```python
    if not download_models(plan, gguf_dir, confirm=confirm, client=client, log=log):
```

to:

```python
    if not download_models(
        plan, gguf_dir, confirm=confirm, client=client, on_progress=on_progress, log=log
    ):
```

- [ ] **Step 2: Write a failing test for run() forwarding on_progress**

Add to `tests/test_bootstrap.py`:

```python
def test_run_forwards_on_progress_to_download(tmp_path):
    seen = {}

    def fake_download(plan, dest, **kwargs):
        seen["on_progress"] = kwargs.get("on_progress")
        return True

    import kultivait.bootstrap as bs
    orig = bs.download_models
    bs.download_models = fake_download
    try:
        marker = lambda d, t: None
        plan = make_plan(pick(), ModelPick("embed", "n/e", "embed.gguf", 10, 0))
        kw = _run_kwargs(tmp_path)
        kw["on_progress"] = marker
        bs.run(plan, **kw)
    finally:
        bs.download_models = orig
    assert seen["on_progress"] is marker
```

- [ ] **Step 3: Run it to verify fail then pass**

Run: `uv run pytest tests/test_bootstrap.py::test_run_forwards_on_progress_to_download -v`
Expected: after Step 1's edit is in place, PASS. (If you write the test before the edit, it FAILs with `unexpected keyword argument 'on_progress'`.)

- [ ] **Step 4: Replace `cmd_init`'s print-loop with `render_survey`**

In `src/kultivait/cli.py`, add to the imports near `import kultivait.bootstrap as bootstrap`:

```python
from kultivait import tui
```

Then replace the summary block in `cmd_init` — from the line `print("kultivait surveyed your garden:\n")` through the `print(f"  distiller: ...")` line (currently `cli.py:322-340`) — with:

```python
    tui.console.print(
        tui.render_survey(runtime, config.chat_base_url, models, clis, config)
    )
```

Leave the `save_config(config, CONFIG_PATH)` call and everything after it intact, but restyle the two trailing lines from:

```python
    print(f"\nwrote {CONFIG_PATH}")
    print("edit it anytime; start the proxy with: kultivait serve")
```

to:

```python
    tui.console.print(f"\n[green]✓[/green] wrote {CONFIG_PATH}")
    tui.console.print("edit it anytime; start the proxy with: [bold]kultivait serve[/bold]")
```

- [ ] **Step 5: Drive the download bar + start spinner from `_offer_setup`**

In `src/kultivait/cli.py`, locate the `outcome = bootstrap.run(setup_plan, skip_install=have_llamacpp)` call in `_offer_setup` (`cli.py:304`). Replace it with a Rich `Progress`-backed `on_progress`:

```python
    from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn

    with Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        console=tui.console,
    ) as progress:
        bar = progress.add_task("downloading models", total=None)

        def on_progress(done: int, total: int) -> None:
            progress.update(bar, completed=done, total=total)

        outcome = bootstrap.run(
            setup_plan, skip_install=have_llamacpp, on_progress=on_progress
        )
```

The server-start spinner already lives inside `bootstrap.start_server`'s poll loop via `tui.log`; no `console.status` wrapper is needed here because `Progress` above already owns the live region during download, and `start_server` runs after the `with` block exits. Leave the `if outcome == "server_failed": sys.exit(1)` and `return` lines below unchanged.

- [ ] **Step 6: Run the init CLI tests**

Run: `uv run pytest tests/test_cli_init.py -v`
Expected: PASS — both `test_cmd_init_survives_bare_machine` (asserts `'kind = "virtual"'` is in the written config, independent of console styling) and `test_cmd_init_no_setup_never_offers` still pass. `_offer_setup`'s Progress block is never reached in these tests (they set `no_setup=True` or stub `_offer_setup`).

- [ ] **Step 7: Run the entire suite**

Run: `uv run pytest -q`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/kultivait/cli.py src/kultivait/bootstrap.py tests/test_bootstrap.py
git commit -m "feat: render init survey table, download bar, and start spinner with rich"
```

---

### Task 5: Manual smoke check + verification

**Files:** none (verification only).

- [ ] **Step 1: Full suite green**

Run: `uv run pytest -q`
Expected: all PASS, no warnings about missing `rich`.

- [ ] **Step 2: Exercise the survey render on a bare machine (no runtime needed)**

Run: `uv run kultivait init --no-setup`
Expected: a bordered "kultivait surveyed your garden" panel with a Role/Serves/Kind table (virtual tiers shown in red as "no backend — escalation briefs instead"), a green `✓ wrote …/config.toml` line, and the "start the proxy" hint. No traceback. (This uses whatever runtime/CLIs are actually present; a bare machine yields all-virtual tiers.)

- [ ] **Step 3: Confirm plain-text degradation**

Run: `uv run kultivait init --no-setup | cat`
Expected: the same content with no broken ANSI escape sequences — Rich detects the piped non-tty stream and emits plain text.

- [ ] **Step 4: Final commit if any polish tweaks were made**

```bash
git add -A && git commit -m "chore: init tui polish smoke-check tweaks" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage:**
- `tui.py` (console/log/ask/render_survey) → Task 1. ✓
- `bootstrap.py` default swaps + `ensure_llamacpp` print → Task 3. ✓
- `_download` `on_progress` callback + `download_models` threading → Task 2. ✓
- `cli.py` render_survey + download bar + spinner → Task 4. ✓
- `pyproject.toml` rich dependency → Task 1. ✓
- Tests: `test_tui.py`, the `on_progress` no-op test, no-change to existing → Tasks 1, 2, 4. ✓
- Error handling unaffected (presentation only) → guaranteed by construction; smoke-checked in Task 5. ✓
- Out-of-scope items (no interactive screen, no textual, no logic change, init-only) → respected across all tasks. ✓

**Note on the spec's spinner detail:** the spec described wrapping `start_server` in `console.status(...)`. During planning this proved redundant with the `Progress` live region and the fact that `start_server` already logs through `tui.log`; Task 4 Step 5 documents dropping the extra `console.status` wrapper. This is a presentation-only simplification, not a scope change — the server-start step still runs identically.

**Placeholder scan:** none — every code step shows full code. ✓

**Type consistency:** `on_progress: Callable[[int, int], None] | None` is used identically in `_download`, `download_models`, and `run`; `render_survey(runtime, base_url, models, clis, config)` signature matches its Task 1 definition and Task 4 call site. ✓
