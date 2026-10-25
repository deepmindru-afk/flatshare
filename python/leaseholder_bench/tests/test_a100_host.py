"""deploy/a100/host.py, a single host's layer over its GPU node's files: its units and drop-ins, its control plane's
configuration, FlashInfer's JIT check, the warm starts and the preflight, against stand-ins for systemctl,
journalctl, nvidia-smi, variant.sh, the engine venv's Python and the engine's endpoints. (profile.py's install path
by the driver, R8, is tested with the rules in test_profile.py.)"""

from __future__ import annotations

import hashlib
import http.server
import importlib.machinery
import importlib.util
import json
import os
import shutil
import socketserver
import stat
import subprocess
import sys
import tempfile
import threading
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest

from leaseholder_bench.e2.run import Host

REPO = Path(__file__).resolve().parents[3]
DEPLOY = REPO / "deploy"
GIB = 1 << 30
A100 = "NVIDIA A100-SXM4-40GB"


def _module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


host_py = _module("lh_a100_host", DEPLOY / "a100" / "host.py")
profile = host_py.profile
lh_svc = _module("lh_svc", DEPLOY / "bin" / "lh-svc")


def gpu(**fields) -> dict:
    return {
        "name": A100,
        "count": 1,
        "memory_mib": 40960,
        "compute_cap": "8.0",
        "driver": "580.65.06",
        "pcie_gen": 4,
        "pcie_width": 16,
        "mig": "Disabled",
        "power_limit_w": 400.0,
        "ecc": "Enabled",
        "numa_node": None,
    } | fields


def facts(**overrides) -> dict:
    """A cloud A100 host's facts: 32 CPUs on one node, 200 GiB of RAM, 100 GiB of /dev/shm, R580, systemd."""
    return {
        "hostname": "a100-vm",
        "kernel": "6.8.0",
        "arch": "x86_64",
        "gpu": gpu(),
        "cpus": {"0": list(range(32))},
        "memory": {"total_bytes": 200 * GIB, "available_bytes": 190 * GIB, "nodes": {}},
        "shm": {"path": "/dev/shm", "total_bytes": 100 * GIB, "free_bytes": 100 * GIB},
        "thp_shmem": "never",
        "container": False,
        "systemd_user": True,
        "linger": True,
        "userns": True,
        "cuda_toolkits": [],
        "cuda_toolkit": None,
        "cuda_compat": None,
        "cc": "/usr/bin/cc",
        "cxx": "/usr/bin/c++",
        "ninja": True,
    } | overrides


def render(tmp_path: Path, host: dict) -> tuple[Path, dict]:
    out = tmp_path / "node"
    node = profile.render(profile.load("a100-40gb"), host, out)
    (out / "host.json").write_text(json.dumps(host))
    return out, node


def script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


def test_staging_renders_before_the_card_is_visible(tmp_path: Path) -> None:
    out, node = render(tmp_path, facts(gpu=None))
    assert node["install"]["path"] == "cu130"
    assert node["values"]["engine.gpu_memory_utilization"] == "0.28"  # the profile's 40,960 MiB
    assert "no GPU visible" in node["warnings"][0]
    assert host_py.arch(host_py.Node.load(out)) == "8.0"


def test_units_link_the_node_and_the_control_plane_with_this_hosts_drop_ins(
    tmp_path: Path, monkeypatch
) -> None:
    out, _ = render(tmp_path, facts(userns=False))
    node = host_py.Node.load(out)
    home = tmp_path / "home"
    units = home / ".config" / "systemd" / "user"
    calls = tmp_path / "calls"
    systemctl = script(tmp_path / "bin" / "systemctl", f'echo "$*" >>{calls}')
    # A link to a unit the repository no longer has, as a tunnel's before lh-tunnel@: it goes.
    units.mkdir(parents=True)
    (units / "lh-tunnel-phastform.service").symlink_to(
        DEPLOY / "cyan" / "systemd" / "lh-tunnel-phastform.service"
    )
    host_py.install_units(node, units, systemctl=str(systemctl))
    assert not (units / "lh-tunnel-phastform.service").is_symlink()
    for name in ("leaseholder.target", "lh-engine.service", "lh-stt.service", "lh-tts.service"):
        assert (units / name).resolve() == (out / "systemd" / name).resolve()
    for name in host_py.CONTROL:
        assert (units / name).resolve() == (DEPLOY / "cyan" / "systemd" / name).resolve()
    worker_variant = units / "lh-voice-worker.service.d" / "variant-vad-turns.conf"
    assert worker_variant.resolve() == (
        DEPLOY / "cyan" / "variants" / "vad-turns" / "lh-voice-worker.conf"
    )
    assert calls.read_text() == "--user daemon-reload\n"
    # The units as lh-svc (and systemd) read them with the drop-ins: no tunnel, the voice worker's STT and TTS on
    # this host's own sockets, the gateway and the voice plane on their cores, the voice plane in the host's
    # namespace.
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    (home / ".config" / "leaseholder").mkdir(parents=True)
    (home / ".config" / "leaseholder" / "livekit.env").write_text(
        "LIVEKIT_URL=ws://127.0.0.1:17880\n"
    )
    plan = node.plan
    assert not any("tunnel" in name for name in host_py.CONTROL)
    assert not list(units.glob("lh-tunnel*"))
    worker = lh_svc.environment(lh_svc.load("lh-voice-worker.service"))
    for server in ("stt", "tts"):
        assert worker[f"LH_{server.upper()}_UDS"] == str(tmp_path / "run" / "lh" / f"{server}.sock")
    assert worker["LH_LLM_UDS"] == str(
        tmp_path / "run" / "lh" / "gateway.sock"
    )  # the gateway, as on cyan
    assert lh_svc.load("lh-gateway.service").last("CPUAffinity") == plan["gateway"]
    for name in ("lh-livekit.service", "lh-voice-worker.service"):
        assert lh_svc.load(name).last("CPUAffinity") == plan["voice"]
    assert lh_svc.load("lh-voicenet.service").values["Environment"] == ["LH_VOICENET=host"]
    assert "LH_TURN_DETECTION=vad" in lh_svc.load("lh-voice-worker.service").values["Environment"]
    # Once namespaces work, the voice plane's drop-in goes.
    node.host["userns"] = True
    host_py.install_units(node, units, systemctl=str(systemctl))
    assert not (units / "lh-voicenet.service.d" / "single-host.conf").exists()
    assert "Environment" not in lh_svc.load("lh-voicenet.service").values


def test_config_writes_the_control_planes_files_once(tmp_path: Path, monkeypatch) -> None:
    out, _ = render(tmp_path, facts(systemd_user=False))
    node = host_py.Node.load(out)
    home = tmp_path / "home"
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    written = host_py.write_config(node, home)
    config = home / ".config" / "leaseholder"
    names = {
        "cache-salt-secret",
        "metrics-key",
        "gateway-keys.env",
        "gateway.toml",
        "livekit.env",
        "e2-host.toml",
    }
    assert {Path(p).name for p in written} == names
    for name in names - {"e2-host.toml"}:
        assert stat.S_IMODE((config / name).stat().st_mode) == 0o600, name
    gateway = tomllib.loads((config / "gateway.toml").read_text())
    run = tmp_path / "run"
    assert gateway["upstream"] == f"unix:{run}/lh/engine.sock"  # the engine's own socket: no tunnel
    assert gateway["cache_salt_secret_file"] == str(config / "cache-salt-secret")
    assert gateway["metrics_key_file"] == str(config / "metrics-key")
    assert gateway["listen"][1] == {
        "unix": str(tmp_path / "run" / "lh" / "gateway.sock"),
        "tenant": "voice",
    }
    keys = host_py._read_env(config / "gateway-keys.env")
    for tenant in ("agents", "bench"):
        key = keys[f"LH_GATEWAY_{tenant.upper()}_KEY"]
        assert gateway["tenants"][tenant]["keys_sha256"] == [
            hashlib.sha256(key.encode()).hexdigest()
        ]
    text = (config / "gateway.toml").read_text()
    assert "<sha256" not in text and "<home>" not in text and "/run/user/1007" not in text
    livekit = host_py._read_env(config / "livekit.env")
    assert (
        livekit["LIVEKIT_KEYS"] == f"{livekit['LIVEKIT_API_KEY']}: {livekit['LIVEKIT_API_SECRET']}"
    )
    # lh-bench e2 run on this host: the engine's and the gateway's sockets, the gateway's TCP listener for AIPerf,
    # the core plan's cores, the profile, the preflight.
    host = Host.load(home)
    plan = node.plan
    assert (host.engine_url, host.gateway_socket, host.gateway_url) == (
        f"unix:{run}/lh/engine.sock",
        f"unix:{run}/lh/gateway.sock",
        "http://127.0.0.1:18080",
    )
    assert (host.aiperf_cores, host.caller_cores) == (plan["aiperf"], plan["callers"])
    assert host.profile == "a100-40gb" and host.preflight == str(out / "preflight.json")
    # A second run keeps every secret and rewrites only e2-host.toml.
    before = {name: (config / name).read_text() for name in names}
    assert [Path(p).name for p in host_py.write_config(node, home)] == ["e2-host.toml"]
    assert {name: (config / name).read_text() for name in names} == before


FAKE_PYTHON = """#!/usr/bin/env python3
# The engine venv's Python: FlashInfer's sampler build (jit-check) and the torch probe (preflight --gpu). They run
# with the engine unit's environment alone, so a file beside it, fail, says which one fails.
import json, os, sys
from pathlib import Path
code = sys.argv[2]
fail = Path(sys.argv[0]).with_name("fail")
fail = fail.read_text() if fail.exists() else ""
if "gen_sampling_module" in code:
    if "build" in fail:
        sys.exit("ld: cannot find -lcudart")
    arch = os.environ["FLASHINFER_CUDA_ARCH_LIST"].replace(".", "")
    ws = os.environ["FLASHINFER_WORKSPACE_BASE"] + "/.cache/flashinfer/0.7.0.post1/" + arch
    print(json.dumps({"library": ws + "/cached_ops/sampling/sampling.so", "workspace": ws, "build_s": 1.5,
                      "env": {k: os.environ.get(k) for k in ("CUDA_HOME", "FLASHINFER_EXTRA_LDFLAGS", "PATH")}}))
elif "torch.cuda.init" in code:
    if "torch" in fail:
        sys.exit("RuntimeError: CUDA driver error: system has unsupported display driver / cuda driver combination (803)")
    print("13.0 NVIDIA A100-SXM4-40GB (8, 0)")
"""


@pytest.fixture
def venv(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    python = home / "leaseholder" / "venvs" / "engine" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(FAKE_PYTHON.replace("/usr/bin/env python3", sys.executable))
    python.chmod(0o755)
    return home


def test_jit_check_builds_with_the_engine_units_environment(
    tmp_path: Path, venv: Path, monkeypatch
) -> None:
    out, _ = render(tmp_path, facts(gpu=None))
    node = host_py.Node.load(out)
    result = host_py.jit_check(node, venv)
    assert (
        result["ok"] and result["arch"] == "8.0"
    )  # the profile's card, before the card is visible
    assert result["cuda_home"] == str(venv) + profile.VENV_CUDA.removeprefix("%h")
    assert result["library"].startswith(str(venv / "leaseholder" / "cache" / "flashinfer"))
    assert result["env"]["FLASHINFER_EXTRA_LDFLAGS"] == f"-L{venv}/leaseholder/cache/cuda_lib"
    assert json.loads((venv / "leaseholder" / "cache" / "jit-check.json").read_text()) == result
    (venv / "leaseholder" / "venvs" / "engine" / "bin" / "fail").write_text("build")
    out, _ = render(tmp_path / "card", facts())
    failed = host_py.jit_check(host_py.Node.load(out), venv)
    assert not failed["ok"] and "cannot find -lcudart" in failed["error"]


class Engine(http.server.BaseHTTPRequestHandler):
    """/server_info and /metrics as vLLM 0.32.0 answers them, and the gateway's /health."""

    def do_GET(self) -> None:
        if self.path.startswith("/server_info"):
            body = json.dumps(
                {
                    "vllm_config": {
                        "scheduler_config": {"async_scheduling": True},
                        "kernel_config": {"linear_backend": "auto"},
                    }
                }
            )
        elif self.path == "/metrics":
            body = 'vllm:cache_config_info{block_size="16",cache_dtype="auto",num_gpu_blocks="2275"} 1.0\n'
        elif self.path == "/health":
            body = ""
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def engine() -> Iterator[str]:
    """The gateway's TCP listener (its /health)."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Engine)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


class UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


@pytest.fixture
def engine_socket() -> Iterator[str]:
    """The engine on its own socket, as the GPU units serve it."""
    # A short path: a socket's must fit sun_path, which a test's tmp_path may not.
    path = Path(tempfile.mkdtemp(prefix="lh-a100-")) / "engine.sock"
    server = UnixServer(str(path), Engine)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"unix:{path}"
    server.shutdown()


JOURNAL = """\
INFO [fp8.py:101] Selected MarlinFP8ScaledMMLinearKernel for Fp8LinearMethod
WARNING [marlin_utils_fp8.py:60] Your GPU does not have native support for FP8 computation but FP8 quantization is being used. Weight-only FP8 compression will be used leveraging the Marlin kernel.
INFO [cuda.py:506] Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER']
INFO [flash_attn.py:1117] Using FlashAttention version 2
INFO [kv_connector.py:40] Using LBHNC KV cache layout.
INFO [topk_topp_sampler.py:120] Using FlashInfer for top-p & top-k sampling.
INFO [core.py:300] init engine (profile, create kv cache, warmup model) took 61.20 seconds
INFO [api_server.py:900] Route: /v1/chat/completions
"""


def test_warm_starts_each_configuration_and_keeps_what_its_start_showed(
    tmp_path: Path, engine_socket: str, monkeypatch
) -> None:
    out, _ = render(tmp_path, facts())
    node = host_py.Node.load(out)
    home = tmp_path / "home"
    calls = tmp_path / "variant.calls"
    # variant.sh fails the 2,048-token cap's start; journalctl shows the engine's selection lines.
    script(
        out / "variant.sh",
        f'echo "$*" >>{calls}\ncase "$*" in *2048*) echo "lh-engine failed" >&2; exit 1 ;; esac',
    )
    journal = tmp_path / "journal.txt"
    journal.write_text(JOURNAL)
    script(tmp_path / "bin" / "journalctl", f"cat {journal}")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    record = host_py.warm(node, list(host_py.WARM), home, engine=engine_socket)
    assert calls.read_text().splitlines() == [
        "measure on",  # /server_info is the measurement drop-in's
        "use frozen",
        "use frozen --max-num-batched-tokens 2048 --long-prefill-token-threshold 0",
        "use frozen --max-num-batched-tokens 1536 --long-prefill-token-threshold 768",
    ]
    frozen = record["configurations"]["frozen"]
    assert frozen["ok"] and frozen["block_size"] == 16 and frozen["num_gpu_blocks"] == 2275
    assert frozen["async_scheduling"] is True and frozen["linear_backend"] == "auto"
    assert "Route:" not in "\n".join(frozen["log"]) and len(frozen["log"]) == 7
    failed = record["configurations"]["2048-0"]
    assert not failed["ok"] and "lh-engine failed" in failed["error"]
    assert json.loads((home / "leaseholder" / "cache" / "warm.json").read_text()) == record
    # A file of configurations, one a line.
    configs = tmp_path / "warm.conf"
    configs.write_text("# C'\n14b qwen3-14b --max-num-seqs 64\n")
    assert host_py.configurations(configs) == [("14b", "qwen3-14b", ("--max-num-seqs", "64"))]


def write_data(data: Path) -> None:
    """The campaign's data as push.sh copies it: two traces of weka-x32, one of them in weka-x32-long, and the
    callers' manifest and two clips, all files."""
    for name, traces in (("weka-x32", (1, 2)), ("weka-x32-long", (2,))):
        (data / name).mkdir(parents=True)
        for n in traces:
            (data / name / f"trace-{n}.json").write_text(
                json.dumps({"requests": [{"in": 1000 * n}]})
            )
    (data / "callers").mkdir()
    (data / "callers" / "manifest.json").write_text(
        json.dumps({"hello": {"kind": "turn"}, "wait": {"kind": "barge_in"}})
    )
    for name in ("hello", "wait"):
        (data / "callers" / f"{name}.wav").write_bytes(b"RIFF")


def ready_host(tmp_path: Path, home: Path, engine: str, monkeypatch, host: dict) -> host_py.Node:
    """A node whose host has run every stage: warm starts, the JIT check, the models, data and binaries, the
    control plane up."""
    out, _ = render(tmp_path, host)
    root = home / "leaseholder"
    cache = root / "cache"
    cache.mkdir(parents=True)
    (cache / "jit-check.json").write_text(
        json.dumps({"ok": True, "arch": "8.0", "seconds": 114.0, "cuda_home": profile.VENV_CUDA})
    )
    entry = {
        "ok": True,
        "ready_s": 95.0,
        "log": JOURNAL.splitlines(),
        "async_scheduling": True,
        "block_size": 16,
        "num_gpu_blocks": 2275,
    }
    (cache / "warm.json").write_text(
        json.dumps({"configurations": {name: entry for name, *_ in host_py.WARM}})
    )
    (root / "venvs" / "engine").mkdir(parents=True, exist_ok=True)
    (root / "venvs" / "engine" / "lh-install-path").write_text("cu130\n")
    hub = tmp_path / "hub"
    for config in ("engine.yaml", "stt.yaml"):
        model, revision = host_py._yaml_model(out / config)
        snapshot = hub / f"models--{model.replace('/', '--')}" / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    write_data(root / "data")
    for name in ("lh-gateway", "livekit-server"):
        script(root / "bin" / name, "echo 1.0")
    procfs = tmp_path / "proc"
    procfs.mkdir()
    (procfs / "mounts").write_text("/dev/nvme0n1p1 / ext4 rw 0 0\ntmpfs /dev/shm tmpfs rw 0 0\n")
    monkeypatch.setenv("LH_PROCFS", str(procfs))
    script(tmp_path / "bin" / "systemctl", "exit 0")
    script(tmp_path / "bin" / "ss", "exit 0")
    script(
        tmp_path / "bin" / "nvidia-smi",
        'case "$*" in -L) echo "GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-1)" ;; esac',
    )
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(host_py, "_free", lambda path: 100 * GIB)  # the test host's own disk aside
    return host_py.Node.load(out)


def statuses(checks: list) -> dict[str, str]:
    return {check.check: check.status for check in checks}


def test_the_preflight_passes_a_ready_host_and_labels_what_the_card_changes(
    tmp_path: Path, venv: Path, engine: str, monkeypatch
) -> None:
    node = ready_host(tmp_path, venv, engine, monkeypatch, facts())
    checks = host_py.preflight(node, gpu=True, home=venv, gateway_url=engine)
    got = statuses(checks)
    # sm80 runs the FP8 checkpoint through Marlin: degraded by name, with what that changes; the rest is ok.
    precision = next(c for c in checks if c.check == "CC and precision path")
    assert precision.status == "degraded" and "MarlinFP8ScaledMMLinearKernel" in precision.value
    assert "BF16 rate" in precision.reason and precision.effect
    expected_ok = {
        "BF16",
        "Attention backend and KV layout",
        "Driver and install path",
        "torch on the GPU",
        "FlashInfer's JIT",
        "Host compilers",
        "Compile and JIT caches",
        "CPU tier and /dev/shm",
        "Pinned memory",
        "Host RAM",
        "Async scheduling",
        "Block size",
        "Engine starts",
        "MIG",
        "systemd --user and linger",
        "User namespaces",
        "Cores",
        "Control plane",
        "Models",
        "Corpus and callers",
        "Binaries",
    }
    assert {name for name, status in got.items() if status == "ok"} >= expected_ok, got
    assert "refused" not in got.values(), got
    corpus = next(c for c in checks if c.check == "Corpus and callers")
    assert corpus.value == "weka-x32-long 1 traces, weka-x32 2 traces, callers 2 clips"
    out = host_py.write_preflight(node, checks)
    data = json.loads(out.read_text())
    assert data["counts"]["degraded"] >= 1 and len(data["checks"]) == len(checks)


def test_the_corpus_row_refuses_traces_that_do_not_resolve_on_this_host(
    tmp_path: Path, venv: Path, engine: str, monkeypatch
) -> None:
    # weka-x32-long as long_traces.py makes it on a workstation: links to weka-x32's traces by their absolute
    # paths. Copied as links, they dangle on the host, where AIPerf would fail on them at block 4's first agent
    # point; the preflight refuses them, as it does an empty corpus and callers without their manifest.
    node = ready_host(tmp_path, venv, engine, monkeypatch, facts())
    data = venv / "leaseholder" / "data"
    workstation = tmp_path / "workstation" / "data" / "weka-x32"
    workstation.mkdir(parents=True)
    for n in (1, 2):
        (workstation / f"trace-{n}.json").write_text(json.dumps({"requests": [{"in": 1000 * n}]}))
    long = data / "weka-x32-long"
    (long / "trace-2.json").unlink()
    done = subprocess.run(
        [
            sys.executable,
            str(REPO / "benchmarks" / "e2" / "long_traces.py"),
            str(workstation),
            str(long),
            "2",
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr

    def corpus() -> host_py.Check:
        checks = host_py.preflight(node, gpu=True, home=venv, gateway_url=engine)
        return next(c for c in checks if c.check == "Corpus and callers")

    assert corpus().status == "ok"  # the workstation's paths still there
    (tmp_path / "workstation").rename(tmp_path / "elsewhere")
    row = corpus()
    assert row.status == "refused", row
    assert (
        "weka-x32-long: 2 of 2 do not resolve to a readable file (trace-1.json, ...)" in row.value
    )
    assert "push.sh copies them, as files" in row.reason
    for path in long.iterdir():
        path.unlink()
    (data / "callers" / "manifest.json").unlink()
    row = corpus()
    assert row.status == "refused"
    assert "weka-x32-long: no traces (*.json)" in row.value
    assert "callers: 1 of 3 do not resolve to a readable file (manifest.json)" in row.value


def test_the_voice_plane_outside_its_namespace_is_degraded_and_the_audit_needs_ss(
    tmp_path: Path, venv: Path, engine: str, monkeypatch
) -> None:
    # With user namespaces refused the voice plane runs in the host's namespace under either supervisor: smoke.sh's
    # listener audit accepts its listeners there by LH_VOICENET=host (test_deploy.py), so the preflight names the
    # host's public addresses and the way to its own namespace; and the audit needs ss.
    monkeypatch.delenv("LH_SUPERVISOR", raising=False)
    monkeypatch.setattr(host_py, "_public_addresses", lambda: ["203.0.113.7"])
    node = ready_host(tmp_path, venv, engine, monkeypatch, facts(userns=False))
    assert host_py.supervisor(node.host) == "systemd"
    (tmp_path / "bin" / "ss").unlink()
    which = host_py.shutil.which
    monkeypatch.setattr(
        host_py.shutil,
        "which",
        lambda name, *a, **k: None if name == "ss" else which(name, *a, **k),
    )
    checks = {c.check: c for c in host_py.preflight(node, gpu=True, home=venv, gateway_url=engine)}
    userns = checks["User namespaces"]
    assert userns.status == "degraded" and userns.value == "refused"
    assert "203.0.113.7 must drop inbound UDP" in userns.reason
    assert (
        "listener audit accepts its UDP sockets, loopback TCP listeners and file sockets"
        in userns.reason
    )
    assert "kernel.apparmor_restrict_unprivileged_userns=0" in userns.reason
    assert (
        checks["Listener audit"].status == "refused"
        and "iproute2" in checks["Listener audit"].reason
    )


def test_the_override_runs_the_units_under_lh_svc_where_systemd_could(
    tmp_path: Path, venv: Path, engine: str, monkeypatch
) -> None:
    # A host with a user manager and lingering: LH_SUPERVISOR=lh-svc runs the units under lh-svc, and with no user
    # namespaces the voice plane in the host's namespace is the degraded path there too.
    monkeypatch.setenv("LH_SUPERVISOR", "lh-svc")
    monkeypatch.setattr(host_py, "_public_addresses", lambda: ["203.0.113.7"])
    node = ready_host(tmp_path, venv, engine, monkeypatch, facts(userns=False))
    checks = {c.check: c for c in host_py.preflight(node, gpu=True, home=venv, gateway_url=engine)}
    assert not [c.check for c in checks.values() if c.status == "refused"]
    userns = checks["User namespaces"]
    assert userns.status == "degraded"
    assert "203.0.113.7 must drop inbound UDP" in userns.reason
    # The audit tells lh-svc's units by their environment, and the voice plane's listeners by LH_VOICENET=host.
    assert "listener audit accepts" in userns.reason and "cannot attribute" not in userns.reason
    supervisor = checks["systemd --user and linger"]
    assert supervisor.status == "degraded" and "LH_SUPERVISOR=lh-svc" in supervisor.value
    # %t is lh-svc's, not systemd's /run/user/UID.
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    run = tmp_path / "state" / "lh-svc" / "run"
    assert host_py.runtime_dir(node.host) == run
    assert host_py.engine_address(node.host) == f"unix:{run}/lh/engine.sock"
    monkeypatch.setenv("LH_SUPERVISOR", "systemd")
    with pytest.raises(host_py.HostError, match="auto or lh-svc"):
        host_py.supervisor(node.host)


def test_env_sh_is_written_for_lh_svc_with_the_override(tmp_path: Path, monkeypatch) -> None:
    out, _ = render(tmp_path, facts())  # systemd --user and lingering
    node = host_py.Node.load(out)
    home = tmp_path / "home"
    monkeypatch.delenv("LH_SUPERVISOR", raising=False)
    assert "shims" not in host_py.write_env(node, home).read_text()  # systemd runs the units
    monkeypatch.setenv("LH_SUPERVISOR", "lh-svc")
    env_sh = host_py.write_env(node, home)
    text = env_sh.read_text()
    assert f"export PATH={REPO}/deploy/bin/shims:$PATH\n" in text
    assert (
        "export LH_SUPERVISOR=lh-svc\n" in text
    )  # kept for the later stages and the scripts that source it
    # Sourced as the scripts source it, with no runtime directory of the session's: the shims first, lh-svc's %t.
    run = home / ".local" / "state" / "lh-svc" / "run"
    sourced = subprocess.run(
        [
            "bash",
            "-c",
            f'source {env_sh} && echo "$PATH" && echo "$XDG_RUNTIME_DIR" && echo "$LH_SUPERVISOR"',
        ],
        env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    path, runtime, chosen = sourced.stdout.splitlines()
    assert path.split(":")[0] == str(REPO / "deploy" / "bin" / "shims")
    assert (runtime, chosen) == (str(run), "lh-svc")
    assert stat.S_IMODE(run.stat().st_mode) == 0o700
    # A host without a user manager runs lh-svc without the override, which env.sh then does not name.
    out, _ = render(tmp_path / "container", facts(systemd_user=False, linger=None))
    monkeypatch.delenv("LH_SUPERVISOR")
    text = host_py.write_env(host_py.Node.load(out), home).read_text()
    assert "deploy/bin/shims" in text and "LH_SUPERVISOR" not in text


def test_bootstrap_refuses_an_unknown_supervisor_before_it_writes_anything(tmp_path: Path) -> None:
    done = subprocess.run(
        ["bash", str(DEPLOY / "a100" / "bootstrap.sh"), "stage"],
        env=os.environ | {"HOME": str(tmp_path), "LH_SUPERVISOR": "systemd"},
        capture_output=True,
        text=True,
    )
    assert done.returncode == 2 and "LH_SUPERVISOR is auto or lh-svc" in done.stderr
    assert not (tmp_path / "leaseholder").exists()


def test_bootstrap_sourced_defines_its_stages_steps_and_runs_nothing(tmp_path: Path) -> None:
    # As the tests of its own flow use it: sourced, bootstrap.sh defines its functions and writes nothing.
    done = subprocess.run(
        [
            "bash",
            "-c",
            f"source {DEPLOY / 'a100' / 'bootstrap.sh'} && declare -F | awk '{{print $3}}' && echo src=$src",
        ],
        env=os.environ | {"HOME": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    defined = set(done.stdout.split())
    assert {
        "main",
        "choose_supervisor",
        "tools",
        "render",
        "shell_env",
        "detect",
        "venvs",
    } <= defined
    assert f"src={REPO}" in done.stdout.splitlines()
    assert not list(tmp_path.iterdir())


def test_the_preflight_names_what_a_bare_container_lacks(
    tmp_path: Path, venv: Path, engine: str, monkeypatch
) -> None:
    host = facts(
        gpu=gpu(mig="Enabled"),
        systemd_user=False,
        linger=None,
        userns=False,
        container=True,
        cc=None,
        shm={"path": "/dev/shm", "total_bytes": 14 * GIB, "free_bytes": 14 * GIB},
    )
    node = ready_host(tmp_path, venv, engine, monkeypatch, host)
    root = venv / "leaseholder"
    (root / "cache" / "jit-check.json").write_text(
        json.dumps(
            {
                "ok": False,
                "arch": "8.0",
                "seconds": 3.0,
                "cuda_home": "x",
                "error": "ld: cannot find -lcudart",
            }
        )
    )
    warm = json.loads((root / "cache" / "warm.json").read_text())
    warm["configurations"]["2048-0"] = {
        "ok": False,
        "ready_s": 30.0,
        "log": [
            "ValueError: Free memory on device (5.1/39.4 GiB) on startup is less than desired GPU memory utilization"
        ],
        "error": "lh-engine failed",
    }
    (root / "cache" / "warm.json").write_text(json.dumps(warm))
    (Path(os.environ["LH_PROCFS"]) / "mounts").write_text("tmpfs / tmpfs rw 0 0\n")
    script(
        Path(os.environ["PATH"].split(os.pathsep)[0]) / "nvidia-smi",
        'case "$*" in -L) printf "GPU 0: A100\\n  MIG 3g.20gb Device 0\\n  MIG 4g.20gb Device 1\\n" ;; esac',
    )
    script(Path(os.environ["PATH"].split(os.pathsep)[0]) / "systemctl", "exit 3")
    shutil.rmtree(root / "data" / "weka-x32")
    (venv / "leaseholder" / "venvs" / "engine" / "bin" / "fail").write_text("torch")
    monkeypatch.setenv("HOME", str(venv))
    checks = host_py.preflight(node, gpu=True, home=venv, gateway_url=engine)
    got = statuses(checks)
    assert got["Host compilers"] == "refused"
    assert got["FlashInfer's JIT"] == "refused"
    assert got["torch on the GPU"] == "refused"
    assert got["MIG"] == "refused"
    assert got["Corpus and callers"] == "refused"
    engine_starts = next(c for c in checks if c.check == "Engine starts")
    assert engine_starts.status == "refused" and "free GPU memory" in engine_starts.reason
    assert got["Compile and JIT caches"] == "degraded"  # tmpfs: the caches go with a reboot
    assert got["CPU tier and /dev/shm"] == "degraded"  # 12 GiB: 2.4 times the pool
    assert got["systemd --user and linger"] == "degraded" and got["User namespaces"] == "degraded"
    assert got["Control plane"] == "pending"
    assert host_py.main(["preflight", str(node.dir)]) == 1  # a refused check fails the stage


def test_the_preflight_waits_for_the_first_start(tmp_path: Path, venv: Path, monkeypatch) -> None:
    out, _ = render(tmp_path, facts(gpu=None))
    node = host_py.Node.load(out)
    monkeypatch.setenv("PATH", f"{tmp_path / 'nothing'}{os.pathsep}/usr/bin{os.pathsep}/bin")
    got = statuses(host_py.preflight(node, home=venv))
    for name in (
        "CC and precision path",
        "Attention backend and KV layout",
        "Pinned memory",
        "Async scheduling",
        "Block size",
        "Engine starts",
        "FlashInfer's JIT",
        "Driver and install path",
    ):
        assert got[name] == "pending", name
    assert "MIG" not in got  # no card yet
