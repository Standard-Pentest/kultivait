# `kultivait init`: Rich-based visual polish — design

Date: 2026-07-15
Status: approved in brainstorming session

## Problem

`kultivait init` is often a new user's first interaction with kultivait
([[2026-07-14-init-hardware-tuning-design]] added the zero-to-local bootstrap
flow it now drives). Today every step — survey summary, tier table, download
progress, confirmations, server-start wait — is plain sequential
`print()`/`input()`. It works, but reads as a wall of text rather than
something intentional.

## Goal

Restyle `kultivait init`'s existing linear flow (survey → confirm → download
→ launch) with color, panels, tables, a real progress bar, and a spinner,
using the `rich` library. This is presentation-only: no new screens, no
navigation, no persistent full-screen app, and no change to the sequence of
steps or decisions the flow makes.

## Decisions made during brainstorming

| Decision | Choice |
|---|---|
| Motivation | Polish / first impression only — not a response to a UX complaint |
| Interactivity level | Display-only polish; reject both a full interactive review screen and an in-between arrow-key-picker — no persistent event loop |
| Library | `rich` (lightweight, no event loop) over `textual` (full TUI framework) |
| Testability | Every existing injection seam (`log=`, `confirm=`, `input_fn=`, `run_cmd=`, etc.) and every `capsys`-asserted substring must keep working unmodified |
| Download progress | Worth a small internal change (a new optional callback) to get a real animated bar, since the current `\r`-style line isn't asserted on in any test |

## Architecture

A new module holds all Rich-specific rendering; nothing else changes shape:

```
tui.py (new)                  presentation only, no side effects of its own
  ├─ console: Console          shared instance
  ├─ log(*a, **k)              console.print wrapper — same call shape as print()
  ├─ ask(prompt, input_fn=input) -> bool   styled question text, same bool contract as bootstrap.ask
  └─ render_survey(...)        Panel + Table for cmd_init's summary

cli.py            cmd_init's print-loop -> tui.render_survey(...)
                   _offer_setup's advisory prints -> tui.log / console.print

bootstrap.py       default log=print -> log=tui.log
                   default confirm=ask -> confirm=tui.ask
                   ensure_llamacpp's bare print() -> tui.console.print()
                   _download gets an optional on_progress(done, total) callback,
                     defaulting to today's \r-text behavior when omitted
                   run()'s call to start_server wrapped in console.status(...) at
                     the call site only — start_server itself is unchanged
```

No function signature loses an existing parameter or changes its contract.
Every test that injects its own `log`/`confirm`/`input_fn`/`run_cmd` fake
keeps working untouched. Tests that rely on the real default and assert via
`capsys` (`test_ensure_llamacpp_without_brew_goes_advisory`,
`test_offer_setup_defers_to_installed_ollama`,
`test_offer_setup_explains_ineligible`, etc.) keep passing because Rich
detects the non-tty pytest stream and drops ANSI codes automatically — the
asserted substrings (`"Homebrew"`, `"ollama serve"`, `"16GB"`) still appear in
plain text.

## Components

1. **`tui.py`**
   - `console = Console()` — module-level singleton, imported wherever needed.
   - `log(*a, **k)` — thin `console.print` wrapper, drop-in default for every
     `log=print` parameter in `bootstrap.py`.
   - `ask(prompt: str, input_fn=input) -> bool` — prints the question via
     `console.print` (bold prompt text, dim `[Y/n]` suffix), then delegates
     the actual yes/no parsing to `input_fn` exactly like today's
     `bootstrap.ask` — same signature, same default-yes behavior.
   - `render_survey(runtime, base_url, models, clis, config)` — builds a
     `Panel` wrapping a `Table` (columns: Role, Serves, Kind), colored
     green/yellow/red for local-free / cloud-billed / no-backend rows.
     Replaces the `for tier in config.tiers: print(...)` loop in `cmd_init`.
     Embed/distiller status lines colored red when `MISSING`, green
     otherwise.

2. **`bootstrap.py`**
   - `download_models`, `run`, `start_server`, `offer_wired_limit`: default
     `log=print` becomes `log=tui.log`.
   - `run`'s default `confirm=ask` becomes `confirm=tui.ask` (bootstrap keeps
     its own plain `ask()` too — untouched, still directly unit-tested — this
     only changes which one is used when no `confirm=` is passed).
   - `ensure_llamacpp`'s bare `print(BREW_INSTALL_HINT)` becomes
     `tui.console.print(BREW_INSTALL_HINT)`.
   - `_download` gains an optional `on_progress: Callable[[int, int], None] |
     None = None` parameter. When `None` (every existing test, and any other
     caller that doesn't pass it), behavior is byte-for-byte what it is
     today — the same `\r`-style text line through `log`. When supplied (by
     the real `cli.py` call path), a `rich.progress.Progress` bar drives it
     instead: one bar per file, advancing on each chunk.
   - `download_models` threads the same optional `on_progress` through to
     `_download` per file.

3. **`cli.py`**
   - `cmd_init`'s summary block becomes one `tui.render_survey(...)` call;
     the final "wrote ~/.kultivait/config.toml" line gets a green checkmark
     style.
   - The real call to `bootstrap.run(...)` (in `_offer_setup`) passes a
     `rich.progress.Progress`-backed `on_progress` and wraps the
     `start_server` step in `console.status("waiting for llama-server...")`
     — a spinner that animates for the duration of the blocking poll loop
     without any change to `start_server` itself.

4. **`pyproject.toml`** — add `rich` to `[project.dependencies]`.

## Testing

- No existing test in `tests/test_bootstrap.py`, `tests/test_cli_init.py`, or
  elsewhere needs to change — every seam and asserted substring is preserved
  by construction (see Architecture).
- New: `tests/test_tui.py` —
  - `render_survey(...)` given a config/model/CLI fixture produces a
    renderable whose plain-text content (via `console.capture()` or
    `Table`/`Panel` internals) contains the expected role/serves/kind
    strings.
  - `tui.ask` calls `input_fn` with the prompt and returns the same
    default-yes boolean semantics as `bootstrap.ask` for `""`, `"y"`, `"N"`.
- New: one test in `tests/test_bootstrap.py` confirming `_download` with
  `on_progress=None` (the default) still produces the current `\r`-style
  text through `log` — i.e. omitting the new parameter is provably a no-op.

## Error handling

Unaffected. Rich rendering is presentation-only, wrapping existing
`log`/`print` call sites; it introduces no new failure modes (Rich degrades
to plain text automatically on non-tty/dumb terminals rather than raising).
All existing error paths (advisory mode, decline paths, checksum mismatch,
download interruption, health-check timeout) keep their current control flow
untouched — only how their messages are painted changes.

## Out of scope

- Any interactive/navigable screen (arrow-key pickers, checklist toggles) —
  explicitly rejected in favor of display-only polish.
- `textual` or any framework with its own event loop / persistent screen.
- Changing the sequence or logic of steps in `cmd_init` or `bootstrap.run`.
- Restyling `kultivait serve`/`route`/`prune`/`escalations`/`harvest` output
  — scoped to `init` only.
