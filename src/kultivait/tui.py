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
    """Drop-in default for bootstrap's `log=print` seams. rich.Console.print
    has no `flush` kwarg (it auto-flushes), so absorb it — bootstrap's
    progress line passes flush=True."""
    kwargs.pop("flush", None)
    console.print(*args, **kwargs)


def ask(prompt: str, input_fn=input) -> bool:
    """[Y/n] confirm, default yes — same contract as bootstrap.ask, styled.

    The styled question is painted via the console; input_fn then reads with
    an empty prompt so the text isn't echoed twice in a real terminal."""
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

    embed_hint = (
        "download a nomic-embed GGUF"
        if runtime == "llamacpp"
        else "run: ollama pull nomic-embed-text"
    )
    embed = config.embed_model or f"[red]MISSING[/red] — {embed_hint}"
    distill = config.distill_model or "[red]MISSING[/red] — pull any 8B+ model"
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
