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


from kultivait.hardware import ModelPick, SetupPlan


class FakeStream:
    def __init__(self, status_code, body: bytes):
        self.status_code = status_code
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        assert self.status_code in (200, 206)

    def iter_bytes(self, chunk_size):
        for i in range(0, len(self._body), 3):  # tiny chunks to exercise the loop
            yield self._body[i : i + 3]


class FakeClient:
    """Serves `body`; honors Range unless ignore_range is set."""

    def __init__(self, body: bytes, ignore_range: bool = False):
        self.body = body
        self.ignore_range = ignore_range
        self.requests = []

    def stream(self, method, url, headers=None, follow_redirects=False):
        headers = dict(headers or {})
        self.requests.append((url, headers))
        if "Range" in headers and not self.ignore_range:
            offset = int(headers["Range"].removeprefix("bytes=").removesuffix("-"))
            return FakeStream(206, self.body[offset:])
        return FakeStream(200, self.body)


def _quiet(*args, **kwargs):
    pass


def pick(name="tiny.gguf", body=b"0123456789"):
    return ModelPick("reasoning", "x/y", name, len(body), 0)


def make_plan(*picks):
    return SetupPlan(eligible=True, reason="test", models=tuple(picks))


def test_download_writes_file_and_clears_part(tmp_path):
    body = b"0123456789"
    client = FakeClient(body)
    bootstrap._download(client, "http://x/tiny.gguf", tmp_path / "tiny.gguf", len(body), log=_quiet)
    assert (tmp_path / "tiny.gguf").read_bytes() == body
    assert not (tmp_path / "tiny.gguf.part").exists()


def test_download_resumes_from_part_with_range_header(tmp_path):
    body = b"0123456789"
    (tmp_path / "tiny.gguf.part").write_bytes(body[:4])
    client = FakeClient(body)
    bootstrap._download(client, "http://x/tiny.gguf", tmp_path / "tiny.gguf", len(body), log=_quiet)
    assert client.requests[0][1]["Range"] == "bytes=4-"
    assert (tmp_path / "tiny.gguf").read_bytes() == body


def test_download_restarts_when_server_ignores_range(tmp_path):
    body = b"0123456789"
    (tmp_path / "tiny.gguf.part").write_bytes(body[:4])
    client = FakeClient(body, ignore_range=True)
    bootstrap._download(client, "http://x/tiny.gguf", tmp_path / "tiny.gguf", len(body), log=_quiet)
    # a 200 despite our Range header means "here's the whole file": no dupes
    assert (tmp_path / "tiny.gguf").read_bytes() == body


def test_download_exact_size_part_renames_without_request(tmp_path):
    body = b"0123456789"
    (tmp_path / "tiny.gguf.part").write_bytes(body)
    client = FakeClient(body)
    ok = bootstrap._download(client, "http://x/tiny.gguf", tmp_path / "tiny.gguf", len(body), log=_quiet)
    assert ok is True
    assert client.requests == []  # no HTTP request needed at all
    assert (tmp_path / "tiny.gguf").read_bytes() == body
    assert not (tmp_path / "tiny.gguf.part").exists()


def test_download_oversized_part_is_dropped_and_restarted(tmp_path):
    body = b"0123456789"
    (tmp_path / "tiny.gguf.part").write_bytes(body + b"garbage-past-the-end")
    client = FakeClient(body)
    ok = bootstrap._download(client, "http://x/tiny.gguf", tmp_path / "tiny.gguf", len(body), log=_quiet)
    assert ok is True
    assert "Range" not in client.requests[0][1]  # started clean, no resume header
    assert (tmp_path / "tiny.gguf").read_bytes() == body


def test_download_size_mismatch_leaves_part_and_reports_failure(tmp_path):
    body = b"0123456789"
    short_body = body[:6]  # connection closed cleanly but early
    client = FakeClient(short_body)
    lines = []

    def log(*args, **kwargs):
        lines.append(" ".join(str(a) for a in args))

    ok = bootstrap._download(client, "http://x/tiny.gguf", tmp_path / "tiny.gguf", len(body), log=log)
    assert ok is False
    assert not (tmp_path / "tiny.gguf").exists()
    assert (tmp_path / "tiny.gguf.part").read_bytes() == short_body
    assert any("incomplete" in line for line in lines)


class RaisingStream(FakeStream):
    """Serves a few bytes then blows up mid-stream, like a dropped connection."""

    def iter_bytes(self, chunk_size):
        yield self._body[:3]
        raise httpx.ReadError("connection dropped")


class RaisingClient:
    """A client whose stream always dies partway through — no clean fakes here."""

    def __init__(self, body: bytes):
        self.body = body
        self.requests = []

    def stream(self, method, url, headers=None, follow_redirects=False):
        self.requests.append((url, dict(headers or {})))
        return RaisingStream(200, self.body)


def test_download_models_interrupted_download_returns_false_and_keeps_part(tmp_path):
    body = b"0123456789" * 10_000
    client = RaisingClient(body)
    ok = bootstrap.download_models(
        make_plan(pick("tiny.gguf", body)), tmp_path, confirm=lambda p: True, client=client, log=_quiet
    )
    assert ok is False
    assert (tmp_path / "tiny.gguf.part").exists()
    assert not (tmp_path / "tiny.gguf").exists()


def test_download_models_skips_complete_files(tmp_path):
    body = b"0123456789"
    (tmp_path / "tiny.gguf").write_bytes(body)
    client = FakeClient(body)
    ok = bootstrap.download_models(
        make_plan(pick()), tmp_path, confirm=_fail_confirm, client=client, log=_quiet
    )
    assert ok is True
    assert client.requests == []


def test_download_models_declined_downloads_nothing(tmp_path):
    client = FakeClient(b"0123456789")
    ok = bootstrap.download_models(
        make_plan(pick()), tmp_path, confirm=lambda p: False, client=client, log=_quiet
    )
    assert ok is False
    assert client.requests == []


def test_download_models_lists_sizes_before_confirming(tmp_path):
    lines = []

    def log(*args, **kwargs):
        lines.append(" ".join(str(a) for a in args))

    prompts = []

    def confirm(prompt):
        prompts.append(prompt)
        return True

    client = FakeClient(b"0123456789")
    bootstrap.download_models(make_plan(pick()), tmp_path, confirm=confirm, client=client, log=log)
    assert any("tiny.gguf" in line for line in lines)
    assert len(prompts) == 1 and "GB" in prompts[0]
    assert (tmp_path / "tiny.gguf").exists()


def full_plan(wired=None):
    return SetupPlan(
        eligible=True,
        reason="test",
        models=(
            ModelPick("simple", "Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf", 10, 78_336),
            ModelPick(
                "embed",
                "nomic-ai/nomic-embed-text-v1.5-GGUF",
                "nomic-embed-text-v1.5.Q8_0.gguf",
                10,
                0,
            ),
        ),
        ctx=16384,
        server_flags=("--jinja", "-fa", "on", "-c", "16384", "--port", "8080"),
        default_gpu_cap_mb=16384,
        wired_limit_mb=wired,
    )


def test_write_artifacts_ini_marks_embedding_model(tmp_path):
    preset, _ = bootstrap.write_artifacts(full_plan(), tmp_path / "home", tmp_path / "ggufs")
    text = preset.read_text()
    assert "[nomic-embed-text-v1.5.Q8_0]" in text
    assert "embedding = 1" in text
    assert str(tmp_path / "ggufs" / "nomic-embed-text-v1.5.Q8_0.gguf") in text


def test_write_artifacts_script_is_executable_with_flags_and_log(tmp_path):
    _, script = bootstrap.write_artifacts(full_plan(), tmp_path / "home", tmp_path / "ggufs")
    text = script.read_text()
    assert text.startswith("#!/bin/sh\n")
    assert "--models-dir" in text and "--models-preset" in text
    assert "-c 16384" in text
    assert "llamacpp.log" in text
    assert script.stat().st_mode & 0o111  # executable


def test_write_artifacts_sysctl_only_as_comment_and_only_when_suggested(tmp_path):
    _, without = bootstrap.write_artifacts(full_plan(), tmp_path / "h1", tmp_path / "g")
    assert "iogpu.wired_limit_mb" not in without.read_text()
    _, with_bump = bootstrap.write_artifacts(full_plan(wired=39936), tmp_path / "h2", tmp_path / "g")
    lines = [l for l in with_bump.read_text().splitlines() if "iogpu.wired_limit_mb" in l]
    assert lines and all(l.startswith("#") for l in lines)
    assert "iogpu.wired_limit_mb=39936" in lines[0].replace(" ", "")


def test_write_artifacts_regenerates_on_rerun(tmp_path):
    home = tmp_path / "home"
    bootstrap.write_artifacts(full_plan(), home, tmp_path / "g")
    plan2 = full_plan(wired=39936)
    _, script = bootstrap.write_artifacts(plan2, home, tmp_path / "g")
    assert "iogpu.wired_limit_mb" in script.read_text()


import httpx


def test_offer_wired_limit_noop_when_plan_fits():
    assert bootstrap.offer_wired_limit(full_plan(), confirm=_fail_confirm, run_cmd=_fail_cmd) is False


def test_offer_wired_limit_declined_runs_nothing():
    ok = bootstrap.offer_wired_limit(
        full_plan(wired=39936), confirm=lambda p: False, run_cmd=_fail_cmd, log=_quiet
    )
    assert ok is False


def test_offer_wired_limit_runs_sysctl_when_accepted():
    calls = []

    def run_cmd(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=0)

    ok = bootstrap.offer_wired_limit(
        full_plan(wired=39936), confirm=lambda p: True, run_cmd=run_cmd, log=_quiet
    )
    assert ok is True
    assert calls == [["sudo", "sysctl", "iogpu.wired_limit_mb=39936"]]


def _script(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    script = home / "start-llamacpp.sh"
    script.write_text("#!/bin/sh\n")
    return script


def test_start_server_polls_until_healthy(tmp_path):
    script = _script(tmp_path)
    popped, attempts = [], iter([httpx.ConnectError("boom"), httpx.ConnectError("boom"), None])

    def http_get(url, timeout=None):
        nxt = next(attempts)
        if nxt:
            raise nxt
        return SimpleNamespace(status_code=200)

    ok = bootstrap.start_server(
        script,
        popen=lambda cmd, **kw: popped.append(cmd),
        http_get=http_get,
        sleep=lambda s: None,
        log=_quiet,
    )
    assert ok is True
    assert popped == [["/bin/sh", str(script)]]


def test_start_server_timeout_tails_log(tmp_path):
    script = _script(tmp_path)
    (script.parent / "llamacpp.log").write_text("line1\nfatal: metal init failed\n")
    lines = []

    def log(*args, **kwargs):
        lines.append(" ".join(str(a) for a in args))

    def http_get(url, timeout=None):
        raise httpx.ConnectError("still down")

    ok = bootstrap.start_server(
        script, popen=lambda cmd, **kw: None, http_get=http_get, sleep=lambda s: None,
        deadline_s=6, log=log,
    )
    assert ok is False
    assert any("metal init failed" in line for line in lines)
    assert any(str(script) in line for line in lines)


def _run_kwargs(tmp_path, **over):
    """run() with everything faked and every step accepted."""
    body = b"0123456789"
    kw = dict(
        home=tmp_path / "home",
        gguf_dir=tmp_path / "ggufs",
        confirm=lambda p: True,
        run_cmd=lambda cmd, **k: SimpleNamespace(returncode=0),
        which=lambda c: f"/opt/homebrew/bin/{c}",
        popen=lambda cmd, **k: None,
        http_get=lambda url, timeout=None: SimpleNamespace(status_code=200),
        sleep=lambda s: None,
        client=FakeClient(body),
        log=_quiet,
    )
    kw.update(over)
    return kw


def test_run_happy_path_creates_everything(tmp_path):
    plan = make_plan(pick(), ModelPick("embed", "n/e", "embed.gguf", 10, 0))
    assert bootstrap.run(plan, **_run_kwargs(tmp_path)) == "ok"
    assert (tmp_path / "ggufs" / "tiny.gguf").exists()
    assert (tmp_path / "home" / "start-llamacpp.sh").exists()
    assert (tmp_path / "home" / "llamacpp-presets.ini").exists()


def test_run_aborts_when_install_declined(tmp_path):
    plan = make_plan(pick(), ModelPick("embed", "n/e", "embed.gguf", 10, 0))
    kw = _run_kwargs(tmp_path, which=lambda c: "/x/brew" if c == "brew" else None,
                     confirm=lambda p: False)
    assert bootstrap.run(plan, **kw) == "aborted"
    assert not (tmp_path / "ggufs").exists()


def test_run_advisory_prints_manual_steps(tmp_path):
    plan = make_plan(pick(), ModelPick("embed", "n/e", "embed.gguf", 10, 0))
    lines = []

    def log(*args, **kwargs):
        lines.append(" ".join(str(a) for a in args))

    kw = _run_kwargs(tmp_path, which=lambda c: None, log=log)
    assert bootstrap.run(plan, **kw) == "aborted"
    assert any("brew install llama.cpp" in line for line in lines)
    assert any("curl" in line and "tiny.gguf" in line for line in lines)


def test_run_reports_server_failure(tmp_path):
    def http_get(url, timeout=None):
        raise httpx.ConnectError("down")

    plan = make_plan(pick(), ModelPick("embed", "n/e", "embed.gguf", 10, 0))
    assert bootstrap.run(plan, **_run_kwargs(tmp_path, http_get=http_get)) == "server_failed"


def test_run_skip_install_never_consults_which(tmp_path):
    def which(c):
        raise AssertionError("which must not be called with skip_install")

    plan = make_plan(pick(), ModelPick("embed", "n/e", "embed.gguf", 10, 0))
    kw = _run_kwargs(tmp_path, which=which)
    kw["skip_install"] = True
    assert bootstrap.run(plan, **kw) == "ok"
