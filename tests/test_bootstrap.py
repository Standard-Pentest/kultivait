"""bootstrap: every side effect is injected — no subprocess, network, sudo,
or real home dir anywhere in these tests."""

from pathlib import Path
from types import SimpleNamespace

import kultivait.bootstrap as bootstrap


def _fail_cmd(*a, **k):  # a run_cmd that must never be reached
    raise AssertionError("run_cmd should not have been called")


def _fail_confirm(prompt):
    raise AssertionError("confirm should not have been called")


def test_ask_defaults_to_yes():
    assert bootstrap.ask("go?", input_fn=lambda _: "") is True
    assert bootstrap.ask("go?", input_fn=lambda _: "y") is True
    assert bootstrap.ask("go?", input_fn=lambda _: "N") is False


def test_ensure_llamacpp_present_short_circuits():
    which = lambda c: "/opt/homebrew/bin/llama-server" if c == "llama-server" else None
    state = bootstrap.ensure_llamacpp(confirm=_fail_confirm, run_cmd=_fail_cmd, which=which)
    assert state == "present"


def test_ensure_llamacpp_without_brew_goes_advisory(capsys):
    state = bootstrap.ensure_llamacpp(
        confirm=_fail_confirm, run_cmd=_fail_cmd, which=lambda c: None
    )
    assert state == "advisory"
    assert "Homebrew" in capsys.readouterr().out


def test_ensure_llamacpp_declined():
    which = lambda c: "/opt/homebrew/bin/brew" if c == "brew" else None
    state = bootstrap.ensure_llamacpp(confirm=lambda p: False, run_cmd=_fail_cmd, which=which)
    assert state == "declined"


def test_ensure_llamacpp_installs_via_brew():
    calls = []

    def run_cmd(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=0)

    which = lambda c: "/opt/homebrew/bin/brew" if c == "brew" else None
    state = bootstrap.ensure_llamacpp(confirm=lambda p: True, run_cmd=run_cmd, which=which)
    assert state == "installed"
    assert calls == [["brew", "install", "llama.cpp"]]


def test_ensure_llamacpp_reports_brew_failure():
    which = lambda c: "/opt/homebrew/bin/brew" if c == "brew" else None
    run_cmd = lambda cmd, **kw: SimpleNamespace(returncode=1)
    state = bootstrap.ensure_llamacpp(confirm=lambda p: True, run_cmd=run_cmd, which=which)
    assert state == "failed"


def test_models_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("KULTIVAIT_LLAMACPP_MODELS_DIR", str(tmp_path / "ggufs"))
    assert bootstrap.models_dir() == tmp_path / "ggufs"


def test_models_dir_default_is_llamacpp_cache(monkeypatch):
    monkeypatch.delenv("KULTIVAIT_LLAMACPP_MODELS_DIR", raising=False)
    monkeypatch.delenv("LLAMA_CACHE", raising=False)
    assert bootstrap.models_dir() == Path.home() / "Library" / "Caches" / "llama.cpp"
