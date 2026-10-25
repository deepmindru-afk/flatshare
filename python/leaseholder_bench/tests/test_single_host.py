"""Single-host mode's machinery (docs/HARDWARE.md): deploy/bin/lh-svc with its systemctl and journalctl shims, which
run the repository's user units where there is no systemd user manager, against stand-in services and the box's own
variant.sh, and deploy/bin/lh-forward, which relays a loopback port to the engine's socket for AIPerf, which speaks
TCP only."""

from __future__ import annotations

import contextlib
import os
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
BIN = REPO / "deploy" / "bin"
SERVICE = """\
import os, signal, sys, time
from pathlib import Path
# A stand-in service: prints its arguments and environment, says it is ready, and runs until it is stopped; with
# STUBBORN it ignores the stop signals.
print("argv", sys.argv[1:], flush=True)
print("env", os.environ.get("GREETING"), os.environ.get("FROM_FILE"), os.environ.get("LH_SVC_UNIT"), flush=True)
if os.environ.get("STUBBORN"):
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, signal.SIG_IGN)
else:
    signal.signal(signal.SIGINT, lambda *_: (print("stopped by SIGINT", flush=True), sys.exit(0)))
Path(os.environ["READY"]).write_text(str(os.getpid()))
while True:
    time.sleep(0.05)
"""
WAIT_READY = "import os, sys, time\nwhile not os.path.exists(sys.argv[1]): time.sleep(0.02)"
NOTIFY = """\
import os, socket, sys, time
from pathlib import Path
# A Type=notify stand-in, as the gateway: READY=1 to NOTIFY_SOCKET a moment after it starts, then it serves.
time.sleep(0.3)
Path(os.environ["MARK"]).write_text("ready")
with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as notify:
    notify.sendto(b"READY=1", os.environ["NOTIFY_SOCKET"])
while True:
    time.sleep(0.05)
"""


@pytest.fixture
def host(tmp_path: Path) -> dict:
    """A home with units in ~/.config/systemd/user, the shims first on PATH, and the services' scripts."""
    home = tmp_path / "home"
    units = home / ".config" / "systemd" / "user"
    units.mkdir(parents=True)
    (home / "svc.py").write_text(SERVICE)
    (home / "wait.py").write_text(WAIT_READY)
    (home / "env-a").write_text('# for svc-a\nFROM_FILE="from the file"\n')
    py = sys.executable
    unit = {
        "svc-a.service": f"""
            [Unit]
            PartOf=box.target
            [Service]
            Type=exec
            ExecStartPre=/bin/mkdir -p %t/lh
            Environment=GREETING=hello READY=%h/a.ready
            EnvironmentFile=%h/env-a
            EnvironmentFile=-%h/missing
            ExecStart={py} %h/svc.py first
            ExecStartPost={py} %h/wait.py %h/a.ready
            KillSignal=SIGINT
            TimeoutStopSec=5
            Restart=on-failure
            RestartSec=0.1
        """,
        "svc-b.service": f"""
            [Unit]
            PartOf=box.target
            After=svc-a.service
            [Service]
            Environment=STUBBORN=1 READY=%h/b.ready
            ExecStart={py} %h/svc.py b
            ExecStartPost={py} %h/wait.py %h/b.ready
            TimeoutStopSec=1
        """,
        "box.target": """
            [Unit]
            Wants=svc-a.service svc-b.service
        """,
        "broken.service": f"""
            [Service]
            ExecStart={py} -c "import sys; print('broken'); sys.exit(3)"
            Restart=on-failure
            RestartSec=0.05
            StartLimitBurst=2
        """,
        "unhealthy.service": f"""
            [Service]
            ExecStart={py} -c "import time; time.sleep(30)"
            ExecStartPost={py} -c "import sys; sys.exit(1)"
        """,
    }
    for name, text in unit.items():
        (units / name).write_text(textwrap.dedent(text).lstrip())
    # A drop-in that replaces svc-a's command and adds flags, which $FLAGS splits into words.
    (units / "svc-a.service.d").mkdir()
    (units / "svc-a.service.d" / "flags.conf").write_text(
        f'[Service]\nEnvironment="FLAGS=--x 1 --y"\nExecStart=\nExecStart={py} %h/svc.py $FLAGS ${{GREETING}}-word\n'
    )
    env = {
        "HOME": str(home),
        "PATH": f"{BIN / 'shims'}{os.pathsep}{os.environ['PATH']}",
        "XDG_RUNTIME_DIR": str(tmp_path / "run"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }
    (tmp_path / "run").mkdir(mode=0o700)
    yield {"home": home, "units": units, "env": env, "tmp": tmp_path}
    subprocess.run(
        ["systemctl", "--user", "stop", "box.target", "broken.service", "unhealthy.service"],
        env=os.environ | env,
        capture_output=True,
    )


def ctl(host: dict, *args: str, tool: str = "systemctl") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [tool, *args], env=os.environ | host["env"], capture_output=True, text=True, timeout=60
    )


def gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_a_target_starts_its_units_in_order_and_stops_every_process(host: dict) -> None:
    before = time.time()
    done = ctl(host, "--user", "start", "box.target")
    assert done.returncode == 0, done.stderr
    assert ctl(host, "--user", "is-active", "svc-a", "svc-b").stdout.split() == ["active"] * 2
    a, b = (int((host["home"] / f"{n}.ready").read_text()) for n in ("a", "b"))
    journal = ctl(
        host, "--user", "-u", "svc-a", "--since", f"@{before}", "-o", "cat", tool="journalctl"
    )
    # The drop-in's command, with $FLAGS split into words and ${GREETING} in place, and the environment's files.
    assert "argv ['--x', '1', '--y', 'hello-word']" in journal.stdout
    # LH_SVC_UNIT names the unit, where systemd's cgroup would (deploy/bin/gpu-guard reads it).
    assert "env hello from the file svc-a.service" in journal.stdout
    assert (Path(host["env"]["XDG_RUNTIME_DIR"]) / "lh").is_dir()  # ExecStartPre, %t
    order = ctl(host, "-u", "svc-a", "-u", "svc-b", "-o", "short-unix", tool="journalctl").stdout
    assert order.index("Started svc-a.service") < order.index("Starting svc-b.service")
    shown = ctl(host, "--user", "show", "svc-a", "--property=Environment", "--value").stdout
    # As lh-bench reads it (Systemd.environment).
    assert shlex.split(shown) == [
        "GREETING=hello",
        f"READY={host['home']}/a.ready",
        "FLAGS=--x 1 --y",
    ]
    # Stopping the target stops both, and returns only once both have exited: svc-b ignores the signal and
    # is killed after its 1 s TimeoutStopSec; svc-a stops on its KillSignal.
    done = ctl(host, "--user", "stop", "box.target")
    assert done.returncode == 0, done.stderr
    assert gone(a) and gone(b)
    assert "stopped by SIGINT" in ctl(host, "-u", "svc-a", "-o", "cat", tool="journalctl").stdout
    inactive = ctl(host, "--user", "is-active", "svc-a", "svc-b")
    assert inactive.returncode == 3 and inactive.stdout.split() == ["inactive"] * 2


def test_a_unit_that_fails_restarts_and_then_fails_its_start(host: dict) -> None:
    assert ctl(host, "--user", "start", "svc-a").returncode == 0
    pid = int((host["home"] / "a.ready").read_text())
    (host["home"] / "a.ready").unlink()
    os.kill(pid, signal.SIGKILL)  # the main process dies: Restart=on-failure starts it again
    deadline = time.time() + 10
    while not (host["home"] / "a.ready").exists() and time.time() < deadline:
        time.sleep(0.05)
    deadline = time.time() + 10
    while ctl(host, "is-active", "svc-a").stdout.strip() != "active" and time.time() < deadline:
        time.sleep(0.05)
    assert int((host["home"] / "a.ready").read_text()) != pid
    assert ctl(host, "is-active", "svc-a").stdout.strip() == "active"
    # A start whose ExecStartPost fails fails, as systemd's does, and leaves no process behind.
    failed = ctl(host, "--user", "start", "unhealthy")
    assert failed.returncode == 1 and "Job for unhealthy.service failed" in failed.stderr
    assert ctl(host, "is-active", "unhealthy").stdout.strip() == "failed"
    # A unit that exits at once has started, as a Type=exec unit has once it runs, then fails after its restarts
    # (StartLimitBurst=2).
    assert ctl(host, "--user", "start", "broken").returncode == 0
    deadline = time.time() + 10
    while ctl(host, "is-active", "broken").stdout.strip() != "failed" and time.time() < deadline:
        time.sleep(0.05)
    assert ctl(host, "is-failed", "broken").returncode == 0
    journal = ctl(host, "-u", "broken", "-o", "cat", tool="journalctl").stdout
    assert journal.count("broken\n") == 2 and "start-limit-hit" in journal  # the burst's two starts
    assert ctl(host, "--user", "reset-failed", "broken").returncode == 0
    assert ctl(host, "is-active", "broken").stdout.strip() == "inactive"


def system_pythons() -> list[str]:
    """The host's own interpreters from 3.10 on, which the shims run before any venv exists (the tests' own
    Python is the venv's)."""
    found = []
    for path in sorted(
        {*Path("/usr/bin").glob("python3"), *Path("/usr/bin").glob("python3.[0-9]*")}
    ):
        if path.name.endswith("-config") or not os.access(path, os.X_OK):
            continue
        done = subprocess.run(
            [str(path), "-c", "import sys; print(sys.version_info >= (3, 10))"],
            capture_output=True,
            text=True,
        )
        if done.stdout.strip() == "True":
            found.append(str(path))
    return found


@pytest.mark.skipif(not system_pythons(), reason="no system python3 of 3.10 or newer in /usr/bin")
def test_the_host_python_runs_the_supervisor_and_the_forwarder(host: dict) -> None:
    # A bare host's python3 runs them, as the shims and the scripts do: Ubuntu 22.04's is 3.10.
    for python in system_pythons():
        done = subprocess.run(
            [python, str(BIN / "lh-svc"), "systemctl", "is-system-running"],
            env=os.environ | host["env"],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0 and done.stdout.strip() == "lh-svc", (python, done.stderr)
        usage = subprocess.run([python, str(BIN / "lh-forward")], capture_output=True, text=True)
        assert usage.returncode == 2 and "lh-forward LISTEN=TARGET" in usage.stderr, (
            python,
            usage.stderr,
        )


def test_the_shims_answer_what_the_scripts_ask(host: dict) -> None:
    assert ctl(host, "--user", "daemon-reload").returncode == 0
    assert ctl(host, "--user", "enable", "-q", "svc-a.service").returncode == 0
    cat = ctl(host, "--user", "cat", "svc-a").stdout
    assert f"# {host['units'] / 'svc-a.service'}" in cat and "flags.conf" in cat
    assert ctl(host, "--user", "start", "nothing").returncode == 5  # no such unit
    assert ctl(host, "--user", "frobnicate").returncode == 2
    assert ctl(host, "-u", "svc-a", "--since", "not a time", tool="journalctl").returncode == 1


def test_a_notify_unit_starts_at_ready_with_its_socket_unit_and_without_a_missing_wanted_one(
    host: dict,
) -> None:
    # As cyan's gateway: Type=notify, Requires= its socket unit, Wants= the tunnel a single host has none of.
    units, py = host["units"], sys.executable
    (host["home"] / "notify.py").write_text(NOTIFY)
    (units / "gw.socket").write_text("[Socket]\nListenStream=%t/lh/gw.sock\n")
    check = f'{py} -c "import pathlib, sys; sys.exit(not pathlib.Path(sys.argv[1]).exists())" %h/gw.ready'
    (units / "gw.service").write_text(
        "[Unit]\nRequires=gw.socket\nWants=lh-tunnel@node.service\nAfter=gw.socket lh-tunnel@node.service\n"
        f"[Service]\nType=notify\nEnvironment=MARK=%h/gw.ready\nExecStart={py} %h/notify.py\n"
        f"ExecStartPost={check}\n"
    )
    sleeper = f"{py} -c 'import time; time.sleep(30)'"
    (units / "needs.service").write_text(
        f"[Unit]\nRequires=absent.service\n[Service]\nExecStart={sleeper}\n"
    )
    (units / "silent.service").write_text(
        f"[Service]\nType=notify\nTimeoutStartSec=1\nExecStart={sleeper}\n"
    )
    try:
        done = ctl(host, "--user", "start", "gw.service")
        assert done.returncode == 0, done.stderr
        # The start returned once READY=1 came, and ExecStartPost ran after it: it found the mark.
        assert ctl(host, "is-active", "gw.socket", "gw.service").stdout.split() == ["active"] * 2
        # A unit that Requires= one that does not exist fails its start, as under systemd.
        missing = ctl(host, "--user", "start", "needs.service")
        assert missing.returncode == 5 and "absent.service" in missing.stderr
        # A notify unit whose main process never says READY=1 fails at its TimeoutStartSec.
        silent = ctl(host, "--user", "start", "silent.service")
        assert (
            silent.returncode == 1 and ctl(host, "is-active", "silent").stdout.strip() == "failed"
        )
        assert "no READY=1" in ctl(host, "-u", "silent", "-o", "cat", tool="journalctl").stdout
    finally:
        ctl(host, "--user", "stop", "gw.service", "gw.socket", "silent.service")
    assert ctl(host, "is-active", "gw.service", "gw.socket").stdout.split() == ["inactive"] * 2


def test_variant_sh_runs_on_the_supervisor(host: dict, tmp_path: Path) -> None:
    # The box's variant.sh with stand-in units: lh-engine runs a make-room drop-in's command, and the flags'
    # drop-in it writes, with the audio servers left running.
    node = tmp_path / "node"
    (node / "variants" / "make-room").mkdir(parents=True)
    shutil.copy(REPO / "deploy" / "phastform" / "variant.sh", node / "variant.sh")
    py, units = sys.executable, host["units"]
    for name in ("lh-stt", "lh-tts", "lh-engine"):
        ready = f"%h/{name}.ready"
        (units / f"{name}.service").write_text(
            f"[Unit]\nPartOf=leaseholder.target\n[Service]\nEnvironment=READY={ready}\nEnvironment=LH_ENGINE_FLAGS=\n"
            f"ExecStart={py} %h/svc.py {name} $LH_ENGINE_FLAGS\nExecStartPost={py} %h/wait.py {ready}\n"
        )
    (units / "leaseholder.target").write_text(
        "[Unit]\nWants=lh-stt.service lh-tts.service lh-engine.service\n"
    )
    (node / "variants" / "make-room" / "lh-engine.conf").write_text(
        f"[Service]\nExecStart=\nExecStart={py} %h/svc.py make-room $LH_ENGINE_FLAGS\n"
    )
    env = os.environ | host["env"]
    try:
        done = subprocess.run(
            ["bash", str(node / "variant.sh"), "use", "make-room", "--budget", "48"],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert done.returncode == 0, done.stdout + done.stderr
        engine = ctl(host, "-u", "lh-engine", "-o", "cat", tool="journalctl").stdout
        assert "argv ['make-room', '--budget', '48']" in engine
        stt = int((host["home"] / "lh-stt.ready").read_text())
        again = subprocess.run(
            ["bash", str(node / "variant.sh"), "use", "frozen"],
            env=env,
            capture_output=True,
            text=True,
        )
        assert again.returncode == 0, again.stdout + again.stderr
        assert (
            "argv ['lh-engine']"
            in ctl(host, "-u", "lh-engine", "-o", "cat", tool="journalctl").stdout
        )
        # Only the engine's drop-ins changed, so the audio servers kept running.
        assert int((host["home"] / "lh-stt.ready").read_text()) == stt and not gone(stt)
    finally:
        ctl(host, "--user", "stop", "leaseholder.target")


def echo_server() -> tuple[socket.socket, int]:
    server = socket.create_server(("127.0.0.1", 0))

    def serve() -> None:
        while True:
            try:
                connection, _ = server.accept()
            except OSError:
                return
            with connection:
                while data := connection.recv(65536):
                    connection.sendall(data.upper())

    threading.Thread(target=serve, daemon=True).start()
    return server, server.getsockname()[1]


def test_lh_forward_relays_a_socket_to_a_port() -> None:
    server, port = echo_server()
    # A short directory: a socket's path has room for 104 bytes on macOS.
    path = Path(tempfile.mkdtemp(prefix="lhf")) / "lh" / "phastform-stt.sock"
    path.parent.mkdir()
    stale = socket.socket(socket.AF_UNIX)  # a socket left by an earlier run is replaced
    stale.bind(str(path))
    stale.close()
    forward = subprocess.Popen(
        [sys.executable, str(BIN / "lh-forward"), f"{path}=127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert "lh-forward:" in forward.stdout.readline()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        with socket.socket(socket.AF_UNIX) as client:
            client.connect(str(path))
            client.sendall(b"what is the capital of france")
            client.shutdown(socket.SHUT_WR)
            reply = b""
            while data := client.recv(65536):
                reply += data
        assert reply == b"WHAT IS THE CAPITAL OF FRANCE"
    finally:
        forward.send_signal(signal.SIGTERM)
        assert forward.wait(timeout=10) == 0
        server.close()
    assert not path.exists()  # removed when stopped
    path.write_text("not a socket")
    refused = subprocess.run(
        [sys.executable, str(BIN / "lh-forward"), f"{path}=127.0.0.1:{port}"],
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0 and "is not a socket" in refused.stderr
    bad = subprocess.run([sys.executable, str(BIN / "lh-forward"), "nowhere"], capture_output=True)
    assert bad.returncode == 2


def test_lh_forward_relays_a_port_to_a_socket() -> None:
    # As AIPerf reaches the engine for the calibration point: a loopback port to the engine's Unix socket.
    path = Path(tempfile.mkdtemp(prefix="lhf")) / "engine.sock"
    server = socket.socket(socket.AF_UNIX)
    server.bind(str(path))
    server.listen()

    def serve() -> None:
        with contextlib.suppress(OSError):
            connection, _ = server.accept()
            with connection:
                while data := connection.recv(65536):
                    connection.sendall(data[::-1])

    threading.Thread(target=serve, daemon=True).start()
    with socket.socket() as probe:  # a free loopback port
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    forward = subprocess.Popen(
        [sys.executable, str(BIN / "lh-forward"), f"127.0.0.1:{port}=unix:{path}"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert f"127.0.0.1:{port} -> unix:{path}" in forward.stdout.readline()
        with socket.create_connection(("127.0.0.1", port)) as client:
            client.sendall(b"engine")
            client.shutdown(socket.SHUT_WR)
            reply = b""
            while data := client.recv(65536):
                reply += data
        assert reply == b"enigne"
    finally:
        forward.send_signal(signal.SIGTERM)
        assert forward.wait(timeout=10) == 0
        server.close()
    relative = subprocess.run(
        [sys.executable, str(BIN / "lh-forward"), f"127.0.0.1:{port}=unix:engine.sock"],
        capture_output=True,
        text=True,
    )
    assert relative.returncode == 2 and "absolute socket path" in relative.stderr


def test_the_voice_plane_runs_in_the_hosts_namespace_where_there_is_none_of_its_own() -> None:
    run = Path(tempfile.mkdtemp(prefix="lhv"))
    (run / "lh").mkdir()
    pidfile = run / "lh" / "voicenet.pid"
    holder = subprocess.Popen(
        ["bash", str(BIN / "voicenet-holder"), str(pidfile)],
        env=os.environ | {"LH_VOICENET": "host"},
    )
    try:
        deadline = time.time() + 10
        while not (pidfile.exists() and pidfile.read_text().strip()) and time.time() < deadline:
            time.sleep(0.05)
        assert pidfile.read_text().strip() == "host"
        done = subprocess.run(
            ["bash", str(BIN / "in-voicenet"), sys.executable, "-c", "print('the host network')"],
            env=os.environ | {"XDG_RUNTIME_DIR": str(run)},
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0 and done.stdout.strip() == "the host network", done.stderr
    finally:
        holder.terminate()
        holder.wait()


@pytest.mark.skipif(not Path("/proc/self/environ").exists(), reason="reads /proc (Linux)")
def test_gpu_guard_tells_the_supervisors_units_from_foreign_processes(tmp_path: Path) -> None:
    # Two processes on the GPU (a stand-in nvidia-smi lists them): one of lh-svc's units, which carries its
    # LH_SVC_UNIT, and a foreign one. The guard passes the unit and refuses the stranger.
    sleep = [sys.executable, "-c", "import time; time.sleep(30)"]
    ours = subprocess.Popen(sleep, env=os.environ | {"LH_SVC_UNIT": "lh-stt.service"})
    theirs = subprocess.Popen(
        sleep, env={k: v for k, v in os.environ.items() if k != "LH_SVC_UNIT"}
    )
    try:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        smi = bin_dir / "nvidia-smi"

        def guard(*pids: int) -> subprocess.CompletedProcess[str]:
            smi.write_text("#!/bin/sh\n" + "".join(f"echo {pid}\n" for pid in pids))
            smi.chmod(0o755)
            return subprocess.run(
                ["bash", str(BIN / "gpu-guard")],
                env=os.environ | {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"},
                capture_output=True,
                text=True,
            )

        assert guard(ours.pid).returncode == 0
        refused = guard(ours.pid, theirs.pid)
        assert refused.returncode == 1 and f"pid {theirs.pid}" in refused.stderr
    finally:
        for process in (ours, theirs):
            process.kill()
            process.wait()
