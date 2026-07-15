"""Zero-to-local bootstrap: install llama.cpp, download right-sized GGUFs,
write tuned launch artifacts, start the server.

Every step is idempotent (already-satisfied work is skipped, so re-running
`kultivait init` converges) and asks before mutating. All process, network,
and filesystem access is injected so tests never touch the real system.
"""

import hashlib
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

import httpx

from kultivait.hardware import SetupPlan

BREW_INSTALL_HINT = (
    "Homebrew is required to install llama.cpp automatically.\n"
    "Install it first:\n"
    '  /bin/bash -c "$(curl -fsSL '
    'https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"\n'
)


def ask(prompt: str, input_fn=input) -> bool:
    """[Y/n] confirm, default yes — every step was already offered explicitly."""
    return input_fn(f"{prompt} [Y/n] ").strip().lower() in ("", "y", "yes")


def models_dir() -> Path:
    """Same GGUF cache llama-server uses and cli._gguf_dirs() scans."""
    override = os.environ.get("KULTIVAIT_LLAMACPP_MODELS_DIR") or os.environ.get(
        "LLAMA_CACHE"
    )
    if override:
        return Path(override)
    return Path.home() / "Library" / "Caches" / "llama.cpp"


def ensure_llamacpp(confirm=ask, run_cmd=subprocess.run, which=shutil.which) -> str:
    """Idempotent install step: "present" | "advisory" | "declined" |
    "installed" | "failed"."""
    if which("llama-server"):
        return "present"
    if not which("brew"):
        print(BREW_INSTALL_HINT)
        return "advisory"
    if not confirm("Install llama.cpp via Homebrew (brew install llama.cpp)?"):
        return "declined"
    result = run_cmd(["brew", "install", "llama.cpp"])
    return "installed" if result.returncode == 0 else "failed"


CHUNK = 1 << 20


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _promote(part: Path, dest: Path, sha256: str, log=print) -> bool:
    """Rename .part -> final only when the bytes match the pinned hash.

    resolve/main URLs are mutable refs — size alone can't rule out upstream
    substitution or corruption, so content is verified before it's trusted.
    A mismatched .part is deleted: a Range resume can never repair wrong
    bytes of the right length."""
    if sha256 and _sha256_of(part) != sha256:
        part.unlink()
        log(f"  {dest.name}: checksum mismatch — discarded corrupt download")
        return False
    part.rename(dest)
    return True


def _download(
    client, url: str, dest: Path, expected_bytes: int, sha256: str = "",
    on_progress: "Optional[Callable[[int, int], None]]" = None, log=print,
) -> bool:
    """Stream to <dest>.part, resume via Range, verify, rename when complete.

    Returns False instead of raising when the result doesn't check out — a
    mismatched size (resumable .part kept) or checksum (.part discarded) is
    a failed download, not a partial success to silently promote to `dest`.
    Callers (download_models) treat any exception during the transfer itself
    as the same kind of recoverable failure.
    """
    if dest.exists() and dest.stat().st_size == expected_bytes:
        log(f"  {dest.name}: already present")
        return True
    part = dest.with_name(dest.name + ".part")
    if part.exists():
        part_size = part.stat().st_size
        if part_size == expected_bytes:
            # already fully fetched (e.g. a prior run died after the write
            # but before the rename) — verify and promote without a request;
            # on a failed check the .part is gone, so fall through to a
            # clean re-download below
            if _promote(part, dest, sha256, log=log):
                return True
        elif part_size > expected_bytes:
            # can't Range-resume past the expected size; something's wrong
            # with this .part (stale/corrupt) — drop it and start clean
            part.unlink()
    headers, mode = {}, "wb"
    if part.exists():
        headers["Range"] = f"bytes={part.stat().st_size}-"
        mode = "ab"
    with client.stream("GET", url, headers=headers, follow_redirects=True) as r:
        if r.status_code == 200 and mode == "ab":
            mode = "wb"  # server ignored Range: start over rather than duplicate
        r.raise_for_status()
        done = part.stat().st_size if mode == "ab" else 0
        with open(part, mode) as f:
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
    if part.stat().st_size != expected_bytes:
        log(
            f"  {dest.name}: incomplete (got {part.stat().st_size / 2**20:.0f} MB, "
            f"expected {expected_bytes / 2**20:.0f} MB) — leaving .part to resume next run"
        )
        return False
    return _promote(part, dest, sha256, log=log)


def download_models(
    plan: SetupPlan,
    dest: Path,
    confirm=ask,
    client: "httpx.Client | None" = None,
    on_progress: "Optional[Callable[[int, int], None]]" = None,
    log=print,
) -> bool:
    """Confirm once (sizes shown), then fetch whatever isn't already on disk."""
    todo = [
        m
        for m in plan.models
        if not (dest / m.filename).exists()
        or (dest / m.filename).stat().st_size != m.approx_bytes
    ]
    if not todo:
        return True
    log("models to download:")
    for m in todo:
        log(f"  {m.filename}  ({m.approx_bytes / 2**30:.1f} GB)")
    total_gb = sum(m.approx_bytes for m in todo) / 2**30
    if not confirm(f"Download {total_gb:.1f} GB into {dest}?"):
        return False
    dest.mkdir(parents=True, exist_ok=True)
    client = client or httpx.Client(timeout=60)
    for m in todo:
        try:
            if not _download(
                client, m.url(), dest / m.filename, m.approx_bytes,
                sha256=m.sha256, on_progress=on_progress, log=log,
            ):
                return False
        except (httpx.HTTPError, OSError) as exc:
            # a multi-GB fetch over a real network drops sometimes; that's
            # not a bug, it's Tuesday — leave the .part alone and let the
            # next `kultivait init` pick up the Range resume, no traceback
            log(f"\n  {m.filename}: download interrupted ({exc})")
            log("download interrupted — re-run `kultivait init` to resume from where it left off")
            return False
    return True


def write_artifacts(
    plan: SetupPlan, kultivait_home: Path, gguf_dir: Path
) -> "tuple[Path, Path]":
    """Presets INI + start script, regenerated from the plan every run —
    like config.toml, these are decisions made visible, not precious state."""
    kultivait_home.mkdir(parents=True, exist_ok=True)
    embed = next(m for m in plan.models if m.role == "embed")
    preset = kultivait_home / "llamacpp-presets.ini"
    preset.write_text(
        "# generated by kultivait init — regenerate by re-running it\n"
        f"[{embed.filename.removesuffix('.gguf')}]\n"
        f"model = {gguf_dir / embed.filename}\n"
        "embedding = 1\n"
    )
    script = kultivait_home / "start-llamacpp.sh"
    log_path = kultivait_home / "llamacpp.log"
    sysctl_comment = ""
    if plan.wired_limit_mb:
        sysctl_comment = (
            "# Optional: raise the GPU memory cap from "
            f"~{plan.default_gpu_cap_mb} MB (resets on reboot):\n"
            f"#   sudo sysctl iogpu.wired_limit_mb={plan.wired_limit_mb}\n"
        )
    script.write_text(
        "#!/bin/sh\n"
        "# generated by kultivait init — regenerate by re-running it\n"
        f"{sysctl_comment}"
        f'exec llama-server --models-dir "{gguf_dir}" --models-preset "{preset}" \\\n'
        f"  {' '.join(plan.server_flags)} \\\n"
        f'  >> "{log_path}" 2>&1\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return preset, script


# The bootstrap always targets llama-server's default port: it exists for
# machines with no local setup, so KULTIVAIT_LLAMACPP_URL (an existing-setup
# override honored by cli.py) is deliberately ignored here — a running custom
# server would have been detected before the offer was ever made.
HEALTH_URL = "http://localhost:8080/v1/models"


def offer_wired_limit(plan: SetupPlan, confirm=ask, run_cmd=subprocess.run, log=print) -> bool:
    """Only offered when plan() flagged the default GPU cap as too tight;
    consent is explicit twice — our confirm, then sudo's password prompt."""
    if not plan.wired_limit_mb:
        return False
    cmd = ["sudo", "sysctl", f"iogpu.wired_limit_mb={plan.wired_limit_mb}"]
    log(
        f"\nYour models want ~{plan.wired_limit_mb} MB of GPU memory but macOS caps it"
        f" at ~{plan.default_gpu_cap_mb} MB by default.\n"
        f"This raises the cap until reboot:  {' '.join(cmd)}"
    )
    if not confirm("Run it now (sudo will ask for your password)?"):
        return False
    return run_cmd(cmd).returncode == 0


def _tail(path: Path, lines: int = 20) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text().splitlines()[-lines:])


def start_server(
    script: Path,
    popen=subprocess.Popen,
    http_get=httpx.get,
    sleep=time.sleep,
    deadline_s: int = 60,
    log=print,
) -> bool:
    """Launch detached, then poll /v1/models — first model load can be slow."""
    log(f"starting llama-server ({script})...")
    popen(["/bin/sh", str(script)], start_new_session=True)
    waited = 0
    while waited < deadline_s:
        try:
            if http_get(HEALTH_URL, timeout=2).status_code == 200:
                log("llama-server is up")
                return True
        except httpx.HTTPError:
            pass
        sleep(2)
        waited += 2
    log(f"llama-server did not answer within {deadline_s}s; last log lines:")
    log(_tail(script.parent / "llamacpp.log"))
    log(f"start it manually and re-run init:  sh {script}")
    return False


def _print_manual_steps(plan: SetupPlan, gguf_dir: Path, log=print) -> None:
    log("manual setup steps:")
    log("  1. install llama.cpp:  brew install llama.cpp")
    log(f"  2. download models into {gguf_dir}:")
    for m in plan.models:
        log(f"       curl -L -o '{gguf_dir / m.filename}' '{m.url()}'")
    log("  3. re-run `kultivait init` — it picks up wherever you left off")


def run(
    plan: SetupPlan,
    *,
    home: "Path | None" = None,
    gguf_dir: "Path | None" = None,
    confirm=ask,
    run_cmd=subprocess.run,
    which=shutil.which,
    popen=subprocess.Popen,
    http_get=httpx.get,
    sleep=time.sleep,
    client=None,
    log=print,
    skip_install: bool = False,
) -> str:
    """Orchestrate the bootstrap: "ok" (server healthy), "aborted" (user
    declined or advisory — continue init as if nothing happened), or
    "server_failed" (don't survey; nothing is listening)."""
    home = home or Path.home() / ".kultivait"
    gguf_dir = gguf_dir or models_dir()
    if not skip_install:
        state = ensure_llamacpp(confirm=confirm, run_cmd=run_cmd, which=which)
        if state == "advisory":
            _print_manual_steps(plan, gguf_dir, log=log)
            return "aborted"
        if state in ("declined", "failed"):
            return "aborted"
    if not download_models(plan, gguf_dir, confirm=confirm, client=client, log=log):
        return "aborted"
    preset, script = write_artifacts(plan, home, gguf_dir)
    log(f"wrote {preset}")
    log(f"wrote {script}")
    offer_wired_limit(plan, confirm=confirm, run_cmd=run_cmd, log=log)
    return "ok" if start_server(script, popen=popen, http_get=http_get, sleep=sleep, log=log) else "server_failed"
