"""tui: presentation-only helpers. render_survey builds a table; ask keeps
bootstrap.ask's default-yes contract. No real console output asserted —
Rich rendering is captured to plain text."""

from kultivait import tui
from kultivait.config import Config, TierSpec


def _config() -> Config:
    return Config(
        tiers=[
            TierSpec(name="simple", role="simple", kind="llamacpp", model="qwen3-4b"),
            TierSpec(name="architect", role="architect", kind="cli", command=["claude", "-p"]),
            TierSpec(name="docs", role="docs", kind="virtual"),
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
    cfg = Config(
        tiers=_config().tiers,
        chat_base_url="http://localhost:8080",
        embed_model=None,
        distill_model=None,
    )
    out = _plain(tui.render_survey("llamacpp", "http://localhost:8080",
                                   [], [], cfg))
    assert "MISSING" in out
    assert "download a nomic-embed GGUF" in out   # llamacpp embed hint
    assert "pull any 8B+ model" in out            # distiller hint


def test_render_survey_missing_embed_hint_is_runtime_specific():
    cfg = Config(
        tiers=_config().tiers,
        chat_base_url="http://localhost:11434",
        embed_model=None,
        distill_model=_config().distill_model,
    )
    out = _plain(tui.render_survey("ollama", "http://localhost:11434", [], [], cfg))
    assert "ollama pull nomic-embed-text" in out   # ollama-specific remediation


def test_ask_default_yes_contract_matches_bootstrap():
    assert tui.ask("go?", input_fn=lambda _: "") is True
    assert tui.ask("go?", input_fn=lambda _: "y") is True
    assert tui.ask("go?", input_fn=lambda _: "N") is False


def test_ask_paints_prompt_to_console_not_input_fn():
    """The styled prompt is displayed via the console; input_fn reads silently
    (empty prompt) so a real terminal doesn't echo the text twice."""
    seen = []
    with tui.console.capture() as cap:
        tui.ask("proceed?", input_fn=lambda p: seen.append(p) or "y")
    assert "proceed?" in cap.get()   # painted by the console
    assert seen == [""]              # input_fn read with no prompt of its own


def test_log_absorbs_flush_kwarg():
    """bootstrap's download progress line calls log(msg, end="", flush=True);
    Console.print has no flush param, so log must swallow it without raising."""
    with tui.console.capture() as cap:
        tui.log("downloading", end="", flush=True)
    assert "downloading" in cap.get()
