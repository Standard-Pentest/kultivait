"""Zero-to-local bootstrap: install llama.cpp, download right-sized GGUFs,
write tuned launch artifacts, start the server.

Every step is idempotent (already-satisfied work is skipped, so re-running
`kultivait init` converges) and asks before mutating. All process, network,
and filesystem access is injected so tests never touch the real system.
"""

import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

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
