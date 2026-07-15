"""Hardware survey: can this machine grow a local garden?

`scan()` parses sysctl output into a HardwareProfile and `plan()` (Task 2)
turns a profile into a SetupPlan — both pure, so any stranger's laptop is a
unit-test fixture, the same trick as config.detect(). The only subprocess
access lives in _read_sysctl().
"""

import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareProfile:
    platform: str  # sys.platform: "darwin" | "linux" | ...
    chip: str  # "Apple M3 Pro" | "Intel(R) Core(TM) ..." | ""
    is_apple_silicon: bool
    ram_gb: float  # hw.memsize / 2**30; 0.0 when unknown


def _read_sysctl() -> str:
    out = subprocess.run(
        ["sysctl", "machdep.cpu.brand_string", "hw.memsize"],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout


def scan(
    sysctl_text: "str | None" = None, platform: "str | None" = None
) -> HardwareProfile:
    """Pure when given sysctl_text; only a live call (both args None on a
    Mac) shells out."""
    platform = platform or sys.platform
    if platform != "darwin":
        return HardwareProfile(platform=platform, chip="", is_apple_silicon=False, ram_gb=0.0)
    text = sysctl_text if sysctl_text is not None else _read_sysctl()
    chip, ram_gb = "", 0.0
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "machdep.cpu.brand_string":
            chip = value.strip()
        elif key.strip() == "hw.memsize":
            try:
                ram_gb = int(value.strip()) / 2**30
            except ValueError:
                ram_gb = 0.0
    return HardwareProfile(
        platform=platform,
        chip=chip,
        is_apple_silicon=chip.startswith("Apple "),
        ram_gb=ram_gb,
    )


MIN_RAM_GB = 24.0
OS_RESERVE_MB = 8192  # leave >=8GB of unified memory for macOS
MARGIN_MB = 2048  # activations + slack on top of weights + KV
CTX_LADDER = [32768, 16384, 8192]
HF_BASE = "https://huggingface.co"


@dataclass(frozen=True)
class ModelPick:
    role: str  # "simple" | "reasoning" | "embed"
    hf_repo: str
    filename: str
    approx_bytes: int  # exact size on HF; also the "already downloaded" check
    kv_bytes_per_token: int  # q8_0 K+V per token; 0 for embedding models

    def url(self) -> str:
        return f"{HF_BASE}/{self.hf_repo}/resolve/main/{self.filename}"


EMBED_PICK = ModelPick(
    role="embed",
    hf_repo="nomic-ai/nomic-embed-text-v1.5-GGUF",
    filename="nomic-embed-text-v1.5.Q8_0.gguf",
    approx_bytes=146_146_432,
    kv_bytes_per_token=0,
)

# kv_bytes_per_token = 2 (K+V) x 8 kv-heads x 128 head-dim x n_layers x
# 1.0625 (q8_0 bytes/elem): Qwen3 4B/8B have 36 layers, 14B has 40, 32B has 64.
QWEN3_4B = ModelPick("simple", "Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf", 2_497_280_256, 78_336)
QWEN3_8B = ModelPick("simple", "Qwen/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf", 5_027_783_488, 78_336)
QWEN3_14B = ModelPick("reasoning", "Qwen/Qwen3-14B-GGUF", "Qwen3-14B-Q4_K_M.gguf", 9_001_752_960, 87_040)
QWEN3_32B_Q4 = ModelPick("reasoning", "Qwen/Qwen3-32B-GGUF", "Qwen3-32B-Q4_K_M.gguf", 19_762_149_024, 139_264)
QWEN3_32B_Q5 = ModelPick("reasoning", "Qwen/Qwen3-32B-GGUF", "Qwen3-32B-Q5_K_M.gguf", 23_214_831_232, 139_264)

# (min_ram_gb, simple pick, reasoning pick, ctx) — first row whose floor the
# machine clears wins, so keep this sorted largest-first.
MODEL_TABLE = [
    (64, QWEN3_8B, QWEN3_32B_Q5, 32768),
    (48, QWEN3_4B, QWEN3_32B_Q4, 32768),
    (32, QWEN3_4B, QWEN3_14B, 32768),
    (24, QWEN3_4B, QWEN3_14B, 16384),
]

assert all(row[3] in CTX_LADDER for row in MODEL_TABLE), (
    "every MODEL_TABLE row's ctx must be a value plan() can step down to"
)
assert MIN_RAM_GB >= MODEL_TABLE[-1][0], (
    "MIN_RAM_GB must be >= the smallest MODEL_TABLE row's floor, or "
    "plan()'s `next(r for r in MODEL_TABLE if profile.ram_gb >= r[0])` "
    "can raise StopIteration for a profile plan() already deemed eligible"
)


@dataclass(frozen=True)
class SetupPlan:
    eligible: bool
    reason: str  # human-readable: why, or why not
    models: "tuple[ModelPick, ...]" = ()
    ctx: int = 0
    server_flags: "tuple[str, ...]" = ()
    default_gpu_cap_mb: int = 0
    wired_limit_mb: "int | None" = None  # None: default cap suffices


def default_gpu_cap_mb(ram_gb: float) -> int:
    """macOS caps GPU-usable unified memory at ~2/3 of RAM (<=36GB) or ~3/4
    (>36GB); iogpu.wired_limit_mb raises it."""
    ram_mb = int(ram_gb * 1024)
    return ram_mb * 2 // 3 if ram_gb <= 36 else ram_mb * 3 // 4


def _budget_mb(models: "tuple[ModelPick, ...]", ctx: int) -> int:
    weights = sum(m.approx_bytes for m in models)
    kv = max((m.kv_bytes_per_token for m in models), default=0) * ctx
    return (weights + kv) // 2**20 + MARGIN_MB


def plan(profile: HardwareProfile) -> SetupPlan:
    if profile.platform != "darwin":
        return SetupPlan(False, f"local-model setup is macOS-only (this is {profile.platform})")
    if not profile.is_apple_silicon:
        return SetupPlan(
            False, f"needs Apple Silicon; this Mac reports {profile.chip or 'an unknown CPU'}"
        )
    if profile.ram_gb < MIN_RAM_GB:
        return SetupPlan(
            False,
            f"needs >={MIN_RAM_GB:.0f}GB unified RAM; this Mac has {profile.ram_gb:.0f}GB",
        )
    _, simple, reasoning, ctx = next(r for r in MODEL_TABLE if profile.ram_gb >= r[0])
    models = (simple, reasoning, EMBED_PICK)
    hard_cap_mb = int(profile.ram_gb * 1024) - OS_RESERVE_MB
    # the plan must always fit RAM minus the OS reserve: step the context
    # down before ever suggesting a wired-limit bump can't-fit territory
    while _budget_mb(models, ctx) > hard_cap_mb and ctx != CTX_LADDER[-1]:
        ctx = CTX_LADDER[CTX_LADDER.index(ctx) + 1]
    cap = default_gpu_cap_mb(profile.ram_gb)
    budget = _budget_mb(models, ctx)
    wired = min(-(-budget // 1024) * 1024, hard_cap_mb) if budget > cap else None
    flags = (
        "--jinja",
        "-ngl", "99",
        "-fa", "on",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "-c", str(ctx),
        "-b", "2048",
        "-ub", "2048",
        "--port", "8080",
    )
    return SetupPlan(
        eligible=True,
        reason=f"{profile.chip} with {profile.ram_gb:.0f}GB unified RAM",
        models=models,
        ctx=ctx,
        server_flags=flags,
        default_gpu_cap_mb=cap,
        wired_limit_mb=wired,
    )
