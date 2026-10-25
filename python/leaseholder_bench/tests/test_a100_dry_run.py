"""The A100 run's core on a CPU, end to end (docs/HARDWARE.md, the core block by block): benchmarks/a100/run.sh runs
blocks 1 to 10 on a single host whose node is rendered from the A100's profile, with the real probes.sh,
calibration.sh (and its lh-forward from a loopback port to the engine's socket), campaign.sh --local and the core's
campaign files, against stand-ins for the engine (its endpoints here, on its Unix socket; its probes' result lines
from fakes/a100/python), lh-bench, AIPerf, nvidia-smi, systemctl and the node's variant.sh. A probe that dies and
an E2 point that fails each stop the run, and each rerun resumes where it stopped, measuring nothing twice; then
pull.sh copies the host's records, and analysis.py reads them back."""

from __future__ import annotations

import http.server
import importlib.util
import json
import os
import shutil
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
FAKES = Path(__file__).parent / "fakes"
RUN = REPO / "benchmarks" / "a100" / "run.sh"
DASH = "\u2013"  # the range dash, as analysis.py writes it
MODEL = "Qwen/Qwen3-4B-Instruct-2507-FP8"
GIB = 1 << 30
FACTS = {
    "hostname": "a100-dry",
    "gpu": {
        "name": "NVIDIA A100-SXM4-40GB",
        "memory_mib": 40960,
        "compute_cap": "8.0",
        "numa_node": None,
        "driver": "580.65.06",
    },
    "cpus": {"0": list(range(32))},
    "memory": {"total_bytes": 200 * GIB, "available_bytes": 190 * GIB, "nodes": {}},
    "shm": {"path": "/dev/shm", "total_bytes": 100 * GIB, "free_bytes": 100 * GIB},
    "cuda_toolkit": None,
}
"""An A100 host's facts, as host.py's detect gives them (test_profile.py's host())."""
PR = "make-room --additional-config.make_room.reserve_tokens 0 --additional-config.make_room.budget_cap_tokens 48"
S2048 = "frozen --max-num-batched-tokens 2048 --long-prefill-token-threshold 0"
S1536 = "frozen --max-num-batched-tokens 1536 --long-prefill-token-threshold 768"
pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("curl") is None,
    reason="the run's scripts need bash and curl",
)


def _module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


profile = _module("lh_profile", REPO / "deploy" / "profile.py")
analysis = _module("lh_a100_analysis", REPO / "benchmarks" / "a100" / "analysis.py")


class Engine(http.server.BaseHTTPRequestHandler):
    """The engine's /v1/models, /metrics and /health, as the run's scripts read them."""

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": MODEL, "max_model_len": 32768}]})
        elif self.path == "/metrics":
            body = "vllm:num_requests_running 0.0\nvllm:num_requests_waiting 0.0\n"
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


class UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


@pytest.fixture
def runtime() -> Iterator[Path]:
    """XDG_RUNTIME_DIR, with the engine on lh/engine.sock as lh-engine.service serves it."""
    # A short path: a socket's must fit sun_path, which a test's tmp_path may not.
    run = Path(tempfile.mkdtemp(prefix="lh-dry-"))
    (run / "lh").mkdir(mode=0o700)
    server = UnixServer(str(run / "lh" / "engine.sock"), Engine)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield run
    server.shutdown()
    shutil.rmtree(run, ignore_errors=True)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def a100_host(hosts: Path) -> Path:
    """The single host's home: the node rendered from a100-40gb with the stand-in variant.sh, the venvs' stand-ins,
    the corpus and the host settings host.py config writes."""
    home = hosts / "localhost"
    node = home / "leaseholder" / "node"
    profile.render(profile.load("a100-40gb"), FACTS, node)
    shutil.copy(FAKES / "campaign" / "variant.sh", node / "variant.sh")
    (home / "leaseholder" / "src").mkdir(parents=True)
    (home / "leaseholder" / "src" / "DEPLOYED").write_text("v0.3-dry\n")
    engine_bin = home / "leaseholder" / "venvs" / "engine" / "bin"
    engine_bin.mkdir(parents=True)
    shutil.copy(FAKES / "a100" / "python", engine_bin / "python")
    bench_bin = home / "leaseholder" / "venvs" / "bench" / "bin"
    bench_bin.mkdir(parents=True)
    shutil.copy(FAKES / "campaign" / "lh-bench", bench_bin / "lh-bench")
    shutil.copy(FAKES / "a100" / "aiperf", bench_bin / "aiperf")
    (bench_bin / "python").symlink_to(sys.executable)
    (home / "leaseholder" / "data" / "weka-x32").mkdir(parents=True)
    config = home / ".config" / "leaseholder"
    config.mkdir(parents=True)
    (config / "e2-host.toml").write_text('aiperf_cores = "21-27"\ncaller_cores = "28-31"\n')
    return home


def lines(path: Path) -> list[str]:
    return path.read_text().splitlines() if path.exists() else []


def ttft(path: Path) -> int:
    """The voice turn's first token in a probe.py run's output."""
    return analysis.results.parse_probe(path.read_text()).ttft_ms


def test_the_core_runs_in_order_resumes_after_each_stop_and_reads_back(
    tmp_path: Path, runtime: Path
) -> None:
    hosts = tmp_path / "hosts"
    home = a100_host(hosts)
    env = os.environ | {
        "HOME": str(home),
        "PATH": os.pathsep.join([str(FAKES / "a100"), str(FAKES / "campaign"), os.environ["PATH"]]),
        "FAKE_HOSTS": str(hosts),
        "FAKE_LH_BENCH": str(Path(sys.executable).with_name("lh-bench")),
        "FAKE_PYTHON": sys.executable,
        "XDG_RUNTIME_DIR": str(runtime),
        "LH_FORWARD_PORT": str(port := free_port()),
    }
    for name in (
        "FAKE_PROBE_FAIL_AT",
        "FAKE_E2_FAIL",
        "LH_NODE_DIR",
        "LH_PROFILE",
        "LH_E2_RESULTS",
    ):
        env.pop(name, None)

    def run(**extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(RUN), *map(str, range(1, 11))],
            env=env | extra,
            capture_output=True,
            text=True,
            timeout=600,
        )

    runs = home / "leaseholder" / "runs"
    blocks = runs / "a100"

    def done() -> list[int]:
        return sorted(int(p.stem) for p in (blocks / "blocks").glob("*.done"))

    # Block 1 runs 8 probes; the 11th, block 2's third, dies as with its engine, and the run stops after block 1.
    first = run(FAKE_PROBE_FAIL_AT="11")
    assert first.returncode == 1, first.stdout + first.stderr
    assert done() == [1]
    assert sorted(p.name for p in (blocks / "block2").glob("stock*")) == [
        "stock2048-kv-1.txt",
        "stock2048-kv-2.txt",
        "stock2048-kv-3.partial",
    ]
    # The rerun resumes at that probe; round 1's frozen point fails, and the run stops in block 4.
    second = run(FAKE_E2_FAIL="w5-disc-frozen-s24")
    assert second.returncode == 1, second.stdout + second.stderr
    assert done() == [1, 2, 3]
    third = run()
    assert third.returncode == 0, third.stdout + third.stderr
    assert done() == list(range(1, 11))
    assert "run: complete: 1 2 3 4 5 6 7 8 9 10" in (blocks / "blocks.log").read_text()

    # Every probe ran once, and the one that died once more: three blocks of 8, and the attempt. The stand-in's
    # first-token times count the probes in order, so each output is the one its probe first wrote.
    assert len(lines(home / "probe-calls")) == 25
    for block, names in (
        ("block1", ["frozen-kv-1", "frozen-kv-3", "frozen-prefill-3", "frozen-tail-2"]),
        ("block2", ["stock2048-kv-3", "stock2048-tail-2", "stock1536-kv-1", "stock1536-tail-2"]),
        ("block3", ["pr-kv-3", "pr-prefill-3", "pr-tail-2"]),
    ):
        for name in names:
            assert (blocks / block / f"{name}.txt").exists(), name
    assert not list(blocks.glob("block*/*.partial"))
    assert ttft(blocks / "block1" / "frozen-kv-1.txt") == 1650 + 1  # run 1's first probe
    assert (
        ttft(blocks / "block2" / "stock2048-kv-2.txt") == 1650 + 10
    )  # written before the probe that died
    assert (
        ttft(blocks / "block2" / "stock2048-kv-3.txt") == 1650 + 12
    )  # run 2's first probe: the one that died
    assert (
        ttft(blocks / "block3" / "pr-kv-1.txt") == 150 + 18
    )  # make-room's, behind the same agents
    assert json.loads((blocks / "block1" / "frozen-e0" / "summary.json").read_text())["fits"]

    # The engines, in the order the blocks and rounds start them: the probe blocks' four settings; round 1 on block
    # 3's engine; the calibration's frozen engine; then each round in its order, starting on the engine the last
    # round ended on. Each point's switch turns the measurement drop-in on first, as campaign.sh does.
    switches = lines(home / "variant.log")
    engines = [
        s.removeprefix("variant.sh use ") for s in switches if s.startswith("variant.sh use ")
    ]
    assert switches.count("variant.sh measure on") == 16 and len(switches) == len(engines) + 16
    assert engines == [
        "frozen",  # block 1
        S2048,  # block 2, in the run that stopped at its third probe
        S2048,  # block 2 again
        S1536,
        PR,  # block 3
        PR,  # round 1: the PRs' values, voice alone, frozen (the point that failed)
        "frozen",
        "frozen",
        "frozen",  # round 1 again: only its frozen point
        "frozen",  # block 5, the calibration
        *("frozen", "frozen", PR),  # round 2
        *(PR, "frozen", "frozen"),  # round 3
        *("frozen", "frozen", PR),  # round 4
        *(PR, "frozen", "frozen"),  # round 5
    ]
    points = [line for line in lines(home / "lh-bench.log") if line.startswith("lh-bench e2 run ")]
    assert len(points) == 16  # the 15 points, and round 1's frozen point twice
    for n in range(1, 6):
        assert len(list(runs.glob(f"e2/a100-r{n}/*/run.json"))) == 3
        mine = [p for p in points if f"/a100-r{n}/" in p]
        assert mine and all(f"--agents-seed {10 + n}" in p for p in mine)
    complete = '"status": "complete"'
    assert complete in (runs / "e2" / "a100-r1" / "w5-disc-frozen-s24" / "run.json").read_text()

    # The calibration point, as calibration.py reads a run: AIPerf's replay straight to the freshly started engine.
    point = blocks / "calibration-s16"
    assert (point / "before.prom").read_text() and (point / "after.prom").read_text()
    assert len(lines(point / "aiperf" / "profile_export.jsonl")) == 3
    settings = lines(point / "aiperf" / "env.txt")
    for setting in (
        f"LH_BENCH_URL=http://127.0.0.1:{port}/v1/chat/completions",  # lh-forward's end of the socket
        "LH_BENCH_SESSIONS=16",
        "LH_BENCH_DURATION=600",
        "LH_BENCH_GRACE=60",
        "LH_BENCH_THINK_CAP=120",
        "LH_BENCH_SEED=11",
        "LH_BENCH_BURST_STARTS=true",
        f"LH_BENCH_MODEL={MODEL}",
        f"LH_BENCH_DATA={home}/leaseholder/data/weka-x32",
    ):
        assert setting in settings, setting
    assert (
        lines(home / "taskset.log") == ["taskset -c 21-27"] * 2
    )  # the relay and AIPerf, on AIPerf's cores
    assert "127.0.0.1:" + str(port) + " -> unix:" in (point / "forward.log").read_text()
    assert lines(home / "systemctl.log") == [
        "systemctl --user reset-failed lh-engine.service",
        "systemctl --user restart lh-engine.service",
        "systemctl --user stop leaseholder.target lh-engine lh-stt lh-tts",  # block 10
    ]

    # The workstation's copy, and the analysis of it.
    pulled = tmp_path / "pulled"
    pull = subprocess.run(
        ["bash", str(REPO / "deploy" / "a100" / "pull.sh"), "--once", "localhost", str(pulled)],
        env=env | {"LH_SSH": str(FAKES / "campaign" / "ssh")},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert pull.returncode == 0, pull.stdout + pull.stderr
    assert json.loads((pulled / "node" / "node.json").read_text())["profile"] == "a100-40gb"
    assert len(list(pulled.glob("results/e2/a100-r*/*/run.json"))) == 15
    assert (pulled / "runs" / "a100" / "calibration-s16" / "after.prom").exists()
    lh_sim = tmp_path / "no-lh-sim"
    text = analysis.report(
        pulled, tmp_path / "corpus", lh_sim, tmp_path / "report", None
    ).read_text()
    # The calibration point and E0 are there; lh-sim is not.
    assert f"## 1. The step model\n\nNot computed: no lh-sim at {lh_sim}" in text
    assert "## 2. The calibration table\n\nNot computed." in text
    # The probes' cases, each on the engine its block ran: about 1.65 s on stock, about 0.15 s on the PRs' values.
    for row in (
        f"| KV case, stock: the voice turn's wait | 1,652 ms (1,651{DASH}1,653, 3 runs) |",
        f"| KV case, stock at 2,048/0 | 1,660 ms (1,659{DASH}1,662, 3 runs) |",
        f"| KV case, the PRs' values: first token | 169 ms (168{DASH}170, 3 runs) |",
        "| block2/stock1536-kv-1 | kv | 1665 | 1765 |",
    ):
        assert row in text, row
    assert (
        "The calls' first-token p95 beside the agents (tail.py): "
        f"frozen, 512/256 1,858 ms (1,857{DASH}1,858, 2 runs); stock, 2,048/0 1,864 ms (1,863{DASH}1,864, 2 runs); "
        f"stock, 1,536/768 1,866 ms (1,866{DASH}1,867, 2 runs); the PRs' values 344 ms (344{DASH}345, 2 runs)."
    ) in text
    assert "## 4. E2 within the A100\n\nThe measured running share" in text
    assert "- Not measured: no point's summary.json has a running share" in text
