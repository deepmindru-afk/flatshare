#!/usr/bin/env python3
"""profile.py: a GPU node's deployment from a hardware profile, deploy/profiles/NAME.toml.

    profile.py detect [--out host.json] [--peer HOST]
    profile.py render PROFILE [--host host.json] --out DIR
    profile.py check PROFILE [--against DIR]
    profile.py update PROFILE
    profile.py check-vllm DIR [FLAG...]

`detect` writes the host's facts as JSON: the GPU as `nvidia-smi` shows it (name, memory, compute capability, driver,
PCIe link, MIG mode, power limit, NUMA node), the CPUs by NUMA node, RAM, /dev/shm, shmem huge pages, the kernel, a
container or VM, root and sudo, systemd's user manager and lingering, user namespaces, a CUDA toolkit, ninja and uv,
and with --peer the TCP round trip to a control-plane host. `render` writes a node's directory: deploy/templates
filled from the profile (engine.yaml, stt.yaml, the units, the make-room drop-in), the box's scripts and the mps
variant as they are, and node.json, the profile's values resolved with the core plan and its warnings. A profile's
"auto" values come from the host's facts by the rules below; a profile without them, as the frozen box's, needs no
host. `check` renders a profile into a scratch directory and compares it byte for byte with its repository copy
(the box's: deploy/phastform), printing each difference, and `update` writes the rendered files back into it after
a template changes (`make profiles`). When the box's own files change, a change to its units for one, `check` shows
the lines to carry into deploy/templates, and a file added to the box's directory has to be rendered, copied or named
in BOX_ONLY before the tests pass. `check-vllm` parses a rendered engine.yaml and stt.yaml with vLLM's own argument
parser and builds their engine configs, as `vllm serve` would, without starting an engine: run it with the engine
venv's Python on a GPU host.

The rules (docs/HARDWARE.md):
- R2, memory budgets: vLLM refuses a start whose free memory is below gpu_memory_utilization x total, with
  kv-cache-memory set too, so the box's budgets in GiB carry over: utilization = budget / the card's memory, rounded
  up to 0.01.
- R3, the CPU tier: the box's ratio to the pool (20 GiB over 5 GiB), bounded by /dev/shm (less 2 GiB) and by the RAM
  available less what a single host keeps for its other processes, in whole GiB; below the pool's own size it is
  refused, since a tier no larger than the pool only mirrors it.
- R7, the core plan: the engine, the audio servers, the gateway, the voice plane, AIPerf and the callers take their
  shares of the host's CPUs in proportion 8:4:2:8:8:4 (34 CPUs give each its want), the engine on the GPU's NUMA node
  and the audio servers off it where there is another. 17 CPUs, half the wants, is the floor for measuring; below it
  the plan still holds every component, and warns.

Standard library only (Python 3.11 or newer), so that a bare host runs it before any venv exists.
"""

from __future__ import annotations

import argparse
import difflib
import fnmatch
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TEMPLATES = HERE / "templates"
PROFILES = HERE / "profiles"
BOX = HERE / "phastform"

RENDERED = (
    "engine.yaml",
    "stt.yaml",
    "systemd/leaseholder.target",
    "systemd/lh-engine.service",
    "systemd/lh-stt.service",
    "systemd/lh-tts.service",
    "variants/make-room/lh-engine.conf",
)
"""The files a node's directory renders from deploy/templates."""
COPIED = (
    "variant.sh",
    "install.sh",
    "measure/lh-engine.conf",
    "systemd/lh-mps.service",
    "variants/mps/lh-engine.conf",
    "variants/mps/lh-stt.conf",
    "variants/mps/lh-tts.conf",
)
"""The files it takes from the box's directory as they are: they name nothing of a node's own."""
BOX_ONLY = {
    "bootstrap.sh": "the box's own setup; another node's is its own",
    "requirements/*": "the engine's and TTS's locks, which a node's bootstrap installs from the source tree",
    "variants/README.md": "the variants' documentation",
    "smoke.sh": "the end-to-end check, which runs from the source tree beside deploy/bin's listener audit and asks "
    "the engine for its model",
    "systemd/lh-engine-proxy.*": "the sglang variant's socket in front of its TCP port, measured on the box only",
    "variants/make-room-v03/*": "W5's control, measured on the box only",
    "variants/pr54530/*": "E2 arm 7a's variant, measured on the box only",
    "variants/sglang/*": "E2 arm 7b's variant, measured on the box only",
}
"""The box's files that a node's directory does not take, and why. Every file of the box's directory is
rendered, copied or here (a test holds it), so that a file another change adds to the box, a unit or a
drop-in, is not silently left out of other nodes: it fails the test until it is placed."""

PLACEHOLDER = re.compile(r"\{\{([a-z0-9_]+\.[a-z0-9_]+)\}\}")
ALONE = re.compile(r"\{\{([a-z0-9_]+\.[a-z0-9_]+)\}\}\n?")

CORES = (("engine", 8), ("audio", 4), ("gateway", 2), ("voice", 8), ("aiperf", 8), ("callers", 4))
"""R7's components and the CPUs each wants. Half of every want is its minimum."""
CORE_FLOOR = sum(want for _, want in CORES) // 2

GIB = 1 << 30
VENV_CUDA = "%h/leaseholder/venvs/engine/lib/python3.12/site-packages/nvidia/cu13"
"""The CUDA 13 toolkit the engine venv carries (nvidia-cuda-nvcc, pinned by the engine lock), Gray's CUDA_HOME."""


class ProfileError(Exception):
    pass


# The host's facts.


def _run(cmd: list[str], timeout: float = 10) -> subprocess.CompletedProcess[str] | None:
    """`cmd`, or None when it is not installed or does not finish."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _read(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


def parse_cpulist(text: str) -> list[int]:
    """The CPUs of a Linux cpulist, such as "0-7,16-23"."""
    cpus: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        first, _, last = part.partition("-")
        cpus.extend(range(int(first), int(last or first) + 1))
    return cpus


def cpulist(cpus: list[int]) -> str:
    """A cpulist of `cpus`, as systemd's CPUAffinity takes it: "0-7,16-23"."""
    parts, run = [], []
    for c in sorted(set(cpus)):
        if run and c == run[-1] + 1:
            run.append(c)
            continue
        if run:
            parts.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else f"{run[0]}")
        run = [c]
    if run:
        parts.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else f"{run[0]}")
    return ",".join(parts)


def _gpu(sysfs: Path) -> dict | None:
    fields = (
        "name,memory.total,compute_cap,driver_version,pcie.link.gen.current,pcie.link.width.current,"
        "mig.mode.current,power.limit,pci.bus_id,ecc.mode.current"
    )
    done = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    if done is None or done.returncode != 0 or not done.stdout.strip():
        return None
    rows = [[x.strip() for x in line.split(",")] for line in done.stdout.strip().splitlines()]
    name, memory, cap, driver, gen, width, mig, power, bus, ecc = rows[0]
    number = lambda x: float(x) if re.fullmatch(r"[0-9.]+", x) else None  # noqa: E731
    # nvidia-smi gives the bus id with an 8-digit domain ("00000000:3B:00.0"), sysfs with a 4-digit one.
    domain, _, rest = bus.partition(":")
    node = _read(sysfs / "bus" / "pci" / "devices" / f"{domain[-4:]}:{rest}".lower() / "numa_node")
    return {
        "name": name,
        "count": len(rows),
        "memory_mib": int(number(memory) or 0),
        "compute_cap": cap,
        "driver": driver,
        "pcie_gen": int(number(gen) or 0),
        "pcie_width": int(number(width) or 0),
        "mig": mig,
        "power_limit_w": number(power),
        "ecc": ecc,
        "bus_id": bus,
        "numa_node": int(node) if node is not None and node.strip().lstrip("-").isdigit() else None,
    }


def _cpus(sysfs: Path) -> dict[str, list[int]]:
    """The CPUs this process may use, by NUMA node."""
    allowed = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    nodes: dict[str, list[int]] = {}
    for path in sorted((sysfs / "devices" / "system" / "node").glob("node[0-9]*")):
        cpus = parse_cpulist(_read(path / "cpulist") or "")
        if allowed is not None and not os.environ.get("LH_SYSFS"):
            cpus = [c for c in cpus if c in allowed]
        if cpus:
            nodes[path.name.removeprefix("node")] = cpus
    if not nodes:
        nodes["0"] = sorted(allowed) if allowed else list(range(os.cpu_count() or 1))
    return nodes


def _meminfo(text: str | None) -> dict[str, int]:
    """meminfo's fields in bytes."""
    out = {}
    for line in (text or "").splitlines():
        m = re.match(r"(?:Node \d+ )?(\w+):\s+(\d+)(?: kB)?", line)
        if m:
            out[m.group(1)] = int(m.group(2)) * 1024
    return out


def _memory(sysfs: Path, procfs: Path) -> dict:
    total = _meminfo(_read(procfs / "meminfo"))
    nodes = {}
    for path in sorted((sysfs / "devices" / "system" / "node").glob("node[0-9]*")):
        info = _meminfo(_read(path / "meminfo"))
        if info:
            nodes[path.name.removeprefix("node")] = {
                "total_bytes": info.get("MemTotal", 0),
                "free_bytes": info.get("MemFree", 0),
            }
    return {
        "total_bytes": total.get("MemTotal", 0),
        "available_bytes": total.get("MemAvailable", 0),
        "nodes": nodes,
    }


def _shm(path: Path) -> dict | None:
    try:
        s = os.statvfs(path)
    except OSError:
        return None
    return {
        "path": str(path),
        "total_bytes": s.f_blocks * s.f_frsize,
        "free_bytes": s.f_bavail * s.f_frsize,
    }


def _bracketed(text: str | None) -> str | None:
    m = re.search(r"\[(\w+)\]", text or "")
    return m.group(1) if m else None


def _cuda_toolkit(local: Path) -> dict | None:
    """The newest CUDA toolkit's nvcc on the PATH or under `local` (/usr/local), with its version."""
    candidates = [shutil.which("nvcc"), *sorted(str(p) for p in local.glob("cuda*/bin/nvcc"))]
    best = None
    for nvcc in filter(None, candidates):
        done = _run([nvcc, "--version"])
        m = re.search(r"release (\d+)\.(\d+)", done.stdout if done else "")
        if m:
            version = (int(m.group(1)), int(m.group(2)))
            if best is None or version > best[0]:
                best = (version, str(Path(nvcc).resolve().parent.parent))
    return {"version": f"{best[0][0]}.{best[0][1]}", "path": best[1]} if best else None


def _userns() -> bool | None:
    """Whether an unprivileged user and network namespace can make a link, as voicenet-holder does."""
    if shutil.which("unshare") is None or shutil.which("ip") is None:
        return None
    done = _run(
        [
            "unshare",
            "--user",
            "--map-root-user",
            "--net",
            "--",
            "sh",
            "-c",
            "ip link add lh0 type dummy && ip link del lh0",
        ]
    )
    return done is not None and done.returncode == 0


def _rtt_ms(host: str, port: int = 22, tries: int = 5) -> list[float] | None:
    """TCP connect times to host:port, in ms."""
    times = []
    for _ in range(tries):
        start = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=3):
                times.append((time.perf_counter() - start) * 1e3)
        except OSError:
            return None
    return times


def detect(peer: str | None = None) -> dict:
    """The host's facts. LH_ROOTFS, LH_SYSFS, LH_PROCFS and LH_SHM put /, /sys, /proc and /dev/shm elsewhere
    (for tests)."""
    root = Path(os.environ.get("LH_ROOTFS", "/"))
    sysfs = Path(os.environ.get("LH_SYSFS", "/sys"))
    procfs = Path(os.environ.get("LH_PROCFS", "/proc"))
    systemd = _run(["systemctl", "--user", "is-system-running"])
    linger = _run(
        ["loginctl", "show-user", os.environ.get("USER", ""), "--property=Linger", "--value"]
    )
    cgroup = _read(procfs / "1" / "cgroup") or ""
    virt = _run(["systemd-detect-virt"])
    sudo = _run(["sudo", "-n", "true"])
    uname = os.uname()
    facts = {
        "hostname": socket.gethostname().split(".")[0],
        "kernel": uname.release,
        "arch": uname.machine,
        "gpu": _gpu(sysfs),
        "cpus": _cpus(sysfs),
        "memory": _memory(sysfs, procfs),
        "shm": _shm(Path(os.environ.get("LH_SHM", "/dev/shm"))),
        "thp_shmem": _bracketed(
            _read(sysfs / "kernel" / "mm" / "transparent_hugepage" / "shmem_enabled")
        ),
        "container": (root / ".dockerenv").exists()
        or (root / "run" / ".containerenv").exists()
        or bool(re.search(r"docker|containerd|kubepods|lxc", cgroup)),
        "virtualization": virt.stdout.strip() if virt is not None else None,
        "root": os.geteuid() == 0,
        "sudo": sudo is not None and sudo.returncode == 0,
        "systemd_user": systemd is not None and systemd.stdout.strip() in ("running", "degraded"),
        "linger": None if linger is None else linger.stdout.strip() == "yes",
        "userns": _userns(),
        "cuda_toolkit": _cuda_toolkit(root / "usr" / "local"),
        "ninja": shutil.which("ninja") is not None,
        "uv": shutil.which("uv")
        or (
            str(Path.home() / ".local/bin/uv") if (Path.home() / ".local/bin/uv").exists() else None
        ),
        "python": sys.version.split()[0],
    }
    if peer:
        facts["peer"] = {"host": peer, "rtt_ms": _rtt_ms(peer)}
    return facts


# The rules.


def utilization(budget_gib: float, memory_mib: int) -> str:
    """R2: `budget_gib` of a card of `memory_mib`, rounded up to 0.01."""
    u = math.ceil(budget_gib / (memory_mib / 1024) * 100 - 1e-9) / 100
    if u > 0.95:
        raise ProfileError(
            f"{budget_gib} GiB is {u:.2f} of a {memory_mib} MiB card: too small a card"
        )
    return f"{u:.2f}"


def cpu_tier(pool_bytes: int, ratio: float, host: dict, reserve_gib: float) -> tuple[int, str]:
    """R3: the CPU tier's bytes, in whole GiB, and what bounds it."""
    want = int(ratio * pool_bytes)
    bounds = [(want, f"{ratio:g} times the GPU pool, as on the box")]
    shm = host.get("shm")
    if shm:
        bounds.append((shm["free_bytes"] - 2 * GIB, "what /dev/shm holds, less 2 GiB"))
    available = host.get("memory", {}).get("available_bytes")
    if available:
        bounds.append(
            (
                available - int(reserve_gib * GIB),
                f"the RAM available less {reserve_gib:g} GiB for the host",
            )
        )
    got, why = min(bounds)
    got = got // GIB * GIB
    if got < pool_bytes:
        raise ProfileError(
            f"refused: a CPU tier of {got / GIB:.0f} GiB ({why}) is below the pool's {pool_bytes / GIB:g} GiB, "
            "so it would only mirror the pool; give /dev/shm and RAM room (docker: --shm-size or --ipc=host)"
        )
    if got < want:
        why = f"{got / pool_bytes:.1f} times the GPU pool, bounded by {why}"
    return got, why


def core_plan(
    nodes: dict[str, list[int]], gpu_node: int | None
) -> tuple[dict[str, str], list[str]]:
    """R7: each component's CPUs and the engine's and the audio servers' NUMA nodes, and the warnings."""
    all_cpus = [c for n in sorted(nodes, key=int) for c in nodes[n]]
    total = len(all_cpus)
    wants = dict(CORES)
    full = sum(wants.values())
    if total >= full:
        shares = dict(wants)
    else:
        # Hamilton's method: each its whole quota, the rest by the largest remainders, and at least one each.
        quotas = {k: total * w / full for k, w in CORES}
        shares = {k: max(1, math.floor(q)) for k, q in quotas.items()}
        for k in sorted(quotas, key=lambda k: quotas[k] - math.floor(quotas[k]), reverse=True):
            if sum(shares.values()) >= total:
                break
            if shares[k] < wants[k]:
                shares[k] += 1
    warnings = []
    if total < CORE_FLOOR:
        warnings.append(
            f"{total} logical CPUs, below the single-host floor of {CORE_FLOOR}: the components share fewer CPUs "
            "than their minimums, so this host checks the mechanics, not the numbers"
        )
    gpu = (
        str(gpu_node)
        if gpu_node is not None and str(gpu_node) in nodes
        else sorted(nodes, key=int)[0]
    )
    others = [n for n in sorted(nodes, key=int) if n != gpu]
    free = {n: list(nodes[n]) for n in nodes}

    def take(count: int, prefer: list[str]) -> list[int]:
        got: list[int] = []
        for n in prefer + [m for m in sorted(nodes, key=int) if m not in prefer]:
            while free[n] and len(got) < count:
                got.append(free[n].pop(0))
        if len(got) < count:  # fewer CPUs than shares: wrap around and share
            got += all_cpus[: count - len(got)]
        return got

    plan, placed = {}, {}
    placed["engine"] = take(shares["engine"], [gpu])
    placed["audio"] = take(shares["audio"], others or [gpu])
    for k in ("gateway", "voice", "aiperf", "callers"):
        placed[k] = take(shares[k], [])
    node_of = {c: n for n, cpus in nodes.items() for c in cpus}
    for k, cpus in placed.items():
        plan[k] = cpulist(cpus)
    plan["engine_numa"] = gpu
    plan["audio_numa"] = node_of[placed["audio"][0]]
    spare = [c for c in all_cpus if not any(c in v for v in placed.values())]
    if spare:
        plan["spare"] = cpulist(spare)
    return plan, warnings


# Profiles.


def load(profile: str) -> dict:
    """A profile by name (deploy/profiles/NAME.toml) or path."""
    path = Path(profile)
    if not path.suffix:
        path = PROFILES / f"{profile}.toml"
    try:
        return tomllib.loads(path.read_text())
    except OSError as e:
        raise ProfileError(f"no profile {profile}: {e}") from None


def flatten(profile: dict) -> dict[str, str]:
    return {
        f"{section}.{key}": str(value)
        for section, table in profile.items()
        if isinstance(table, dict)
        for key, value in table.items()
        if not isinstance(value, dict)
    }


def resolve(profile: dict, host: dict | None) -> tuple[dict[str, str], dict | None, list[str]]:
    """The profile's values with every "auto" one derived from the host's facts, the core plan, and the warnings."""
    values = flatten(profile)
    autos = sorted(k for k, v in values.items() if v == "auto")
    if not autos:
        return values, None, []
    name = values["profile.name"]
    if host is None:
        raise ProfileError(
            f"{name} derives {', '.join(autos)} from the host: pass --host (profile.py detect)"
        )
    gpu = host.get("gpu")
    if not gpu:
        raise ProfileError(f"{name}: the host shows no GPU (nvidia-smi)")
    budgets = profile.get("budgets", {})
    derived: dict[str, str] = {}
    if values.get("profile.node") == "auto":
        values["profile.node"] = host.get("hostname") or "the GPU host"
    if values.get("engine.gpu_memory_utilization") == "auto":
        values["engine.gpu_memory_utilization"] = utilization(
            budgets["engine_gib"], gpu["memory_mib"]
        )
    if values.get("stt.gpu_memory_utilization") == "auto":
        values["stt.gpu_memory_utilization"] = utilization(budgets["stt_gib"], gpu["memory_mib"])
    derived["total_gib"] = f"{gpu['memory_mib'] / 1024:.1f}"
    derived["engine_budget"] = f"{budgets.get('engine_gib', 0):.2f} GiB"
    pool = int(values["engine.kv_cache_memory"])
    if values.get("engine.cpu_bytes_to_use") == "auto":
        tier, why = cpu_tier(pool, budgets["cpu_tier_ratio"], host, budgets["host_reserve_gib"])
        values["engine.cpu_bytes_to_use"] = str(tier)
        derived["cpu_tier_gib"] = f"{tier // GIB}"
        derived["cpu_tier_note"] = why
    toolkit = host.get("cuda_toolkit")
    system = toolkit is not None and tuple(int(x) for x in toolkit["version"].split(".")) >= (13, 0)
    cuda = toolkit["path"] if system else VENV_CUDA
    unit_values = {
        "units.cuda_home": cuda,
        "units.engine_path": f"%h/leaseholder/venvs/engine/bin:{cuda}/bin:/usr/local/bin:/usr/bin:/bin:%h/.local/bin",
        "units.stt_path": f"%h/leaseholder/venvs/engine/bin:{cuda}/bin:/usr/local/bin:/usr/bin:/bin",
        "units.stt_cuda_home": "" if system else f"Environment=CUDA_HOME={cuda}",
    }
    # The middle of a comment sentence: text.cuda ends it.
    derived["cuda_note"] = (
        f"the host's CUDA {toolkit['version']} toolkit at {toolkit['path']} compiles"
        if system
        else "the host has no CUDA 13 toolkit, so the engine JIT-compiles with the CUDA 13 compiler its\n# venv "
        "carries (nvidia-cuda-nvcc, pinned by the engine lock), as on Gray"
    )
    plan, warnings = core_plan({k: v for k, v in host["cpus"].items()}, gpu.get("numa_node"))
    unit_values |= {
        "units.engine_cpus": plan["engine"],
        "units.engine_numa": plan["engine_numa"],
        "units.audio_cpus": plan["audio"],
        "units.audio_numa": plan["audio_numa"],
    }
    for k, v in unit_values.items():
        if values.get(k) == "auto":
            values[k] = v
    several = len(host["cpus"]) > 1
    derived["engine_numa_note"] = (
        f'NUMA node {plan["engine_numa"]} is local to the GPU (detected). "preferred" spills to another node when '
        "it is short."
        if several
        else f"One NUMA node; the engine's CPUs are {plan['engine']} (the core plan in node.json)."
    )
    derived["audio_numa_note"] = (
        f"NUMA node {plan['audio_numa']} (CPUs {plan['audio']}) for the audio servers"
        + (
            "; the GPU-local node is the engine's."
            if plan["audio_numa"] != plan["engine_numa"]
            else "."
        )
    )
    values |= {f"derived.{k}": v for k, v in derived.items()}
    left = sorted(k for k, v in values.items() if v == "auto")
    if left:
        raise ProfileError(f"{name}: no rule derives {', '.join(left)}")
    return values, plan, warnings


def fill(text: str, values: dict[str, str]) -> str:
    """`text` with every {{section.key}} replaced; a line that holds only a placeholder of an empty value goes."""
    try:
        lines = [
            line
            for line in text.splitlines(keepends=True)
            if not ((m := ALONE.fullmatch(line)) and values[m.group(1)] == "")
        ]
        text = "".join(lines)
        for _ in range(4):  # a value may hold placeholders of its own
            text, n = PLACEHOLDER.subn(lambda m: values[m.group(1)], text)
            if n == 0:
                break
    except KeyError as e:
        raise ProfileError(f"no value for the placeholder {e.args[0]}") from None
    if PLACEHOLDER.search(text):
        raise ProfileError(f"placeholders left unfilled: {sorted(set(PLACEHOLDER.findall(text)))}")
    return text


def render(profile: dict, host: dict | None, out: Path) -> dict:
    """Writes the node's directory to `out`; returns node.json's content."""
    values, plan, warnings = resolve(profile, host)
    for rel in RENDERED:
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(fill((TEMPLATES / rel).read_text(), values))
    for rel in COPIED:
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BOX / rel, dest)
    node = {
        "profile": values["profile.name"],
        "values": {k: v for k, v in sorted(values.items()) if not k.startswith("text.")},
        "plan": plan,
        "warnings": warnings,
    }
    (out / "node.json").write_text(json.dumps(node, indent=2) + "\n")
    return node


def check(profile: dict, against: Path) -> list[str]:
    """The files that differ between the profile's rendering and `against`."""
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch)
        render(profile, None, out)
        return [
            rel
            for rel in RENDERED + COPIED
            if not (against / rel).is_file()
            or (out / rel).read_bytes() != (against / rel).read_bytes()
        ]


def differences(profile: dict, against: Path) -> dict[str, str]:
    """Each file that differs between the profile's rendering and `against`, with its unified diff
    (rendered, then `against`): where the box's files changed, the template line to change."""
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch)
        render(profile, None, out)
        diffs = {}
        for rel in check(profile, against):
            theirs = (
                (against / rel).read_text().splitlines(keepends=True)
                if (against / rel).is_file()
                else []
            )
            diffs[rel] = "".join(
                difflib.unified_diff(
                    (out / rel).read_text().splitlines(keepends=True),
                    theirs,
                    f"rendered/{rel}",
                    str(against / rel),
                )
            )
        return diffs


def unplaced(box: Path = BOX) -> list[str]:
    """The files of the box's directory that a node's directory neither renders nor copies and that
    BOX_ONLY does not name: each needs a place before another node runs without it."""
    files = sorted(
        str(path.relative_to(box))
        for path in box.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    return [
        rel
        for rel in files
        if rel not in RENDERED + COPIED
        and not any(fnmatch.fnmatch(rel, pattern) for pattern in BOX_ONLY)
    ]


def update(profile: dict, into: Path) -> list[str]:
    """Writes the profile's rendered files into its repository copy `into`; returns those that changed."""
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch)
        render(profile, None, out)
        changed = []
        for rel in RENDERED:
            new = (out / rel).read_bytes()
            if (into / rel).read_bytes() != new:
                (into / rel).write_bytes(new)
                changed.append(rel)
        return changed


def check_vllm(node_dir: Path, flags: list[str]) -> int:
    """Parses the node's engine.yaml and stt.yaml with vLLM's argument parser and builds their engine configs."""
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.entrypoints.launchers.cli_args import make_arg_parser
        from vllm.usage.usage_lib import UsageContext
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except ImportError as e:
        print(f"profile: check-vllm runs with the engine venv's Python ({e})", file=sys.stderr)
        return 2
    for name, extra in (("engine.yaml", flags), ("stt.yaml", [])):
        parser = make_arg_parser(FlexibleArgumentParser(prog="vllm serve"))
        args = parser.parse_args(["--config", str(node_dir / name), *extra])
        config = AsyncEngineArgs.from_cli_args(args).create_engine_config(
            UsageContext.OPENAI_API_SERVER
        )
        cache = config.cache_config
        print(
            f"{name}: {config.model_config.model}, kv_cache_memory_bytes {cache.kv_cache_memory_bytes}, "
            f"gpu_memory_utilization {cache.gpu_memory_utilization}, "
            f"max_num_batched_tokens {config.scheduler_config.max_num_batched_tokens}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="profile.py", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("detect", help="write the host's facts as JSON")
    p.add_argument("--out", type=Path, help="write them to this file instead of stdout")
    p.add_argument(
        "--peer", help="also time TCP connects to this host's SSH port (split mode's RTT check)"
    )
    p = commands.add_parser("render", help="write a node's directory from a profile")
    p.add_argument("profile")
    p.add_argument(
        "--host",
        type=Path,
        help="the host's facts (profile.py detect), for a profile's auto values",
    )
    p.add_argument("--out", type=Path, required=True)
    p = commands.add_parser(
        "check", help="compare a profile's rendering with its repository copy, byte for byte"
    )
    p.add_argument("profile")
    p.add_argument(
        "--against", type=Path, help="the directory to compare with (default: the profile's dir)"
    )
    p = commands.add_parser(
        "update", help="write a profile's rendered files into its repository copy"
    )
    p.add_argument("profile")
    p = commands.add_parser(
        "check-vllm", help="parse a rendered node's configs with vLLM's own parser"
    )
    p.add_argument("dir", type=Path)
    p.add_argument(
        "flags",
        nargs=argparse.REMAINDER,
        help="engine flags after the config, as a variant adds them",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "detect":
            text = json.dumps(detect(args.peer), indent=2) + "\n"
            if args.out:
                args.out.write_text(text)
            else:
                sys.stdout.write(text)
        elif args.command == "render":
            host = json.loads(args.host.read_text()) if args.host else None
            node = render(load(args.profile), host, args.out)
            for warning in node["warnings"]:
                print(f"profile: warning: {warning}", file=sys.stderr)
            print(f"profile: {node['profile']} rendered into {args.out}")
        elif args.command in ("check", "update"):
            profile = load(args.profile)
            copy = REPO / profile["profile"]["dir"] if profile["profile"].get("dir") else None
            against = getattr(args, "against", None) or copy
            if against is None:
                raise ProfileError(f"{args.profile} has no repository copy: pass --against")
            if args.command == "update":
                for rel in update(profile, against):
                    print(f"profile: wrote {against / rel}")
                return 0
            differ = differences(profile, against)
            for rel, diff in differ.items():
                print(f"profile: {rel} differs from {against / rel}:\n{diff}", file=sys.stderr)
            if differ:
                return 1
            print(f"profile: {args.profile} renders {against} byte for byte")
        else:
            return check_vllm(args.dir, args.flags)
    except ProfileError as e:
        print(f"profile: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
