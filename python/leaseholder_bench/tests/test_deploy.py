"""The deployment: what the GPU nodes serve and the tunnels reach, cyan's install, and the script that
authorizes cyan's tunnel key on a GPU node, run against stand-ins for ssh and systemd."""

from __future__ import annotations

import os
import re
import socketserver
import stat
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEPLOY = REPO / "deploy"
SERVICES = {8000, 8100, 8880}
"""A GPU node's services on its loopback: the engine, STT and TTS."""
KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMa9oQ1uNw4Z5fT1n2x0bq7d1H0mGJ1s7mYcQjv8Yw5k"
"""A tunnel key's type and blob, as ssh-keygen writes them before the comment."""
LOGIN = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIB7m2Qp9xR4tYc3vLk8wNs6fJd0aHe5uPq1zXo2iGb4M me@laptop"
"""Another key in a GPU node's authorized_keys: its user's own."""

FAKE_SSH = """\
#!/bin/sh
# ssh HOST COMMAND...: runs COMMAND in HOST's home under $FAKE_HOSTS, as sshd runs it in the user's shell.
host=$1
shift
cd "$FAKE_HOSTS/$host" && exec sh -c "$*"
"""


def authorize(
    work: Path, public_key: str, authorized_keys: str = ""
) -> tuple[subprocess.CompletedProcess[str], str]:
    """Runs authorize-tunnel.sh from cyan to phastform, hosts kept under ``work``, cyan's tunnel key
    being ``public_key`` and phastform's authorized_keys ``authorized_keys``; returns the run and the
    file it leaves."""
    hosts, bin_dir = work / "hosts", work / "bin"
    for host in ("cyan", "phastform"):
        (hosts / host / ".ssh").mkdir(parents=True)
    (hosts / "cyan" / ".ssh" / "lh_tunnel_ed25519.pub").write_text(f"{public_key}\n")
    authorized = hosts / "phastform" / ".ssh" / "authorized_keys"
    authorized.write_text(authorized_keys)
    bin_dir.mkdir()
    (bin_dir / "fake-ssh").write_text(FAKE_SSH)
    (bin_dir / "fake-ssh").chmod(0o755)
    env = os.environ | {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "FAKE_HOSTS": str(hosts),
        "LH_SSH": "fake-ssh",
    }
    done = subprocess.run(
        ["bash", str(DEPLOY / "authorize-tunnel.sh"), "cyan", "phastform"],
        env=env,
        capture_output=True,
        text=True,
    )
    return done, authorized.read_text()


def permitted(authorized_keys: str) -> set[int]:
    """The loopback ports authorized_keys lets a key forward to."""
    return {int(port) for port in re.findall(r'permitopen="127\.0\.0\.1:(\d+)"', authorized_keys)}


def test_the_gpu_nodes_publish_no_kv_events_and_the_tunnels_reach_only_their_services(
    tmp_path: Path,
) -> None:
    # vLLM's KV-event publisher connects to a TCP endpoint and serves a replay of the events, which carry
    # every cached block's token ids and session id: on a shared host any user could bind or ask for them,
    # and nothing of the deployment reads them.
    engine = (DEPLOY / "phastform" / "engine.yaml").read_text()
    assert "kv-events-config" not in engine and "tcp://" not in engine
    for unit in ("lh-tunnel-phastform", "lh-tunnel-gray"):
        text = (DEPLOY / "cyan" / "systemd" / f"{unit}.service").read_text()
        forwarded = {int(forward.rsplit(":", 1)[1]) for forward in re.findall(r"-L (\S+)", text)}
        assert forwarded == SERVICES, unit
    smoke = (DEPLOY / "phastform" / "smoke.sh").read_text()
    audited = re.search(r"grep -E ':\(([\d|]+)\)\$'", smoke)
    assert audited is not None
    assert {int(port) for port in audited.group(1).split("|")} == SERVICES
    done, authorized = authorize(tmp_path, f"{KEY} lh-tunnel@cyan")
    assert done.returncode == 0, done.stderr
    assert permitted(authorized) == SERVICES


def test_the_tunnel_key_may_forward_to_the_services_and_listen_nowhere(tmp_path: Path) -> None:
    earlier = f'restrict,port-forwarding,permitopen="127.0.0.1:5558" {KEY} lh-tunnel@cyan'
    options = (
        'restrict,port-forwarding,permitopen="127.0.0.1:8000",permitopen="127.0.0.1:8100",'
        'permitopen="127.0.0.1:8880",permitlisten="127.0.0.1:1",command="/bin/false"'
    )
    # The key as ssh-keygen writes it, and without its comment, which is optional.
    for name, public_key in (("commented", f"{KEY} lh-tunnel@cyan"), ("bare", KEY)):
        done, authorized = authorize(tmp_path / name, public_key, f"{LOGIN}\n{earlier}\n")
        # The earlier entry for the key is replaced; every other key stays.
        assert authorized.splitlines() == [LOGIN, f"{options} {public_key}"], name
        assert done.stdout == f"authorized: {public_key.rsplit(' ', 1)[-1]} (1 entry)\n", name
    # A client without a key changes nothing.
    done, authorized = authorize(tmp_path / "empty", "", f"{LOGIN}\n")
    assert done.returncode == 2 and "no public key" in done.stderr
    assert authorized == f"{LOGIN}\n"


def test_cyans_install_makes_the_gateways_secret_and_metrics_key_once(tmp_path: Path) -> None:
    home, bin_dir = tmp_path / "home", tmp_path / "bin"
    home.mkdir()
    bin_dir.mkdir()
    for tool in ("systemctl", "loginctl"):  # cyan's user manager, which the test leaves alone
        (bin_dir / tool).write_text("#!/bin/sh\n")
        (bin_dir / tool).chmod(0o755)
    env = os.environ | {"HOME": str(home), "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    install = ["bash", str(DEPLOY / "cyan" / "install.sh")]
    subprocess.run(install, env=env, check=True, capture_output=True)
    config = home / ".config" / "leaseholder"
    files = [config / "cache-salt-secret", config / "metrics-key"]
    made = [path.read_text() for path in files]
    for path, text in zip(files, made, strict=True):
        assert re.fullmatch(r"[0-9a-f]{64}\n", text), path
        # The gateway refuses a file that another user may open.
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert made[0] != made[1]
    # A new secret would change every tenant's cache salt, a new key every scraper's: the next install keeps them.
    subprocess.run(install, env=env, check=True, capture_output=True)
    assert [path.read_text() for path in files] == made
    assert sorted(p.name for p in config.iterdir()) == ["cache-salt-secret", "metrics-key"]


FAKE_SYSTEMCTL = """\
#!/bin/sh
# systemctl --user ...: a GPU node's user manager, every unit running; logs each call that changes something.
shift
if [ "$1" = is-active ]; then
  shift
  [ "$1" = -q ] || for unit; do echo active; done
  exit 0
fi
echo "$*" >>"$HOME/systemctl.log"
"""


def test_the_engine_serves_its_development_routes_only_with_the_measurement_drop_in(
    tmp_path: Path,
) -> None:
    unit = (DEPLOY / "phastform" / "systemd" / "lh-engine.service").read_text()
    assert "VLLM_SERVER_DEV_MODE" not in unit
    measure = DEPLOY / "phastform" / "measure" / "lh-engine.conf"
    assert "Environment=VLLM_SERVER_DEV_MODE=1" in measure.read_text().splitlines()

    home, bin_dir = tmp_path / "home", tmp_path / "bin"
    home.mkdir()
    bin_dir.mkdir()
    (bin_dir / "systemctl").write_text(FAKE_SYSTEMCTL)
    (bin_dir / "systemctl").chmod(0o755)
    env = os.environ | {"HOME": str(home), "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    drop_in = home / ".config" / "systemd" / "user" / "lh-engine.service.d" / "measure.conf"
    log = home / "systemctl.log"

    def variant(*args: str) -> str:
        done = subprocess.run(
            ["bash", str(DEPLOY / "phastform" / "variant.sh"), *args],
            env=env,
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stderr
        return done.stdout

    # On: linked, and the running engine restarts to take it; asking again changes nothing.
    variant("measure", "on")
    assert drop_in.resolve() == measure
    assert log.read_text().splitlines() == [
        "daemon-reload",
        "reset-failed lh-engine",
        "restart lh-engine",
    ]
    log.unlink()
    assert variant("measure", "on") == "variant: the measurement drop-in is already on\n"
    assert not log.exists()
    # Switching the engine's variant keeps it.
    variant("use", "pr54530")
    assert drop_in.resolve() == measure
    assert (drop_in.parent / "variant-pr54530.conf").exists()
    variant("use", "frozen")
    assert drop_in.resolve() == measure
    # Off: removed, and the engine restarts without the development routes.
    log.unlink()
    variant("measure", "off")
    assert not drop_in.exists()
    assert log.read_text().splitlines() == [
        "daemon-reload",
        "reset-failed lh-engine",
        "restart lh-engine",
    ]


class _Health(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200 if self.path == "/health" else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


class _UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def test_wait_healthy_reaches_a_server_on_its_unix_socket() -> None:
    wait = ["bash", str(DEPLOY / "bin" / "wait-healthy")]
    # A short path: a socket's must fit sun_path, which pytest's tmp_path may not.
    socket = Path(tempfile.mkdtemp(prefix="lh-wait-")) / "engine.sock"
    server = _UnixServer(str(socket), _Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        done = subprocess.run(
            [*wait, "--unix-socket", str(socket), "http://localhost/health", "5"],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.startswith(f"wait-healthy: http://localhost/health on {socket} healthy")
    finally:
        server.shutdown()
    # Nobody serves the socket now: it gives up after its timeout.
    done = subprocess.run(
        [*wait, "--unix-socket", str(socket), "http://localhost/health", "0"],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 1
    assert f"on {socket} not healthy after 0s" in done.stderr


FAKE_SS = """\
#!/bin/sh
# ss -H -lntup | -lxp: the listeners in $FAKE_SS, as ss prints them.
case $2 in
  -lntup) cat "$FAKE_SS/inet" ;;
  -lxp) cat "$FAKE_SS/unix" ;;
esac
"""


class Listeners:
    """A host for audit-listeners: what ``ss`` lists, and the processes' cgroups and commands under a
    fake /proc. Each process of ours is in an lh-*.service unit."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.runtime = root / "run"
        (self.runtime / "lh").mkdir(parents=True)
        (self.runtime / "lh").chmod(0o700)
        self.ss = root / "ss"
        self.ss.mkdir()
        self.proc = root / "proc"
        self.inet: list[str] = []
        self.unix: list[str] = []
        self.pids = 0
        bin_dir = root / "bin"
        bin_dir.mkdir()
        (bin_dir / "ss").write_text(FAKE_SS)
        (bin_dir / "ss").chmod(0o755)
        self.env = os.environ | {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "FAKE_SS": str(self.ss),
            "LH_AUDIT_PROC": str(self.proc),
            "XDG_RUNTIME_DIR": str(self.runtime),
        }

    def process(self, unit: str | None, command: str = "vllm serve") -> str:
        """A process, in ``unit`` or (None) another user's session; returns ss's note of it."""
        self.pids += 1
        pid = str(1000 + self.pids)
        (self.proc / pid).mkdir(parents=True)
        cgroup = (
            f"/user.slice/user-1007.slice/user@1007.service/app.slice/{unit}"
            if unit
            else "/init.scope"
        )
        (self.proc / pid / "cgroup").write_text(f"0::{cgroup}\n")
        (self.proc / pid / "cmdline").write_bytes(command.replace(" ", "\0").encode())
        return f'users:(("python3",pid={pid},fd=3))'

    def tcp(self, address: str, unit: str | None, command: str = "vllm serve") -> None:
        held = self.process(unit, command) if unit else ""  # ss shows no process of another user's
        self.inet.append(f"tcp LISTEN 0 4096 {address} 0.0.0.0:* {held}")

    def socket(self, path: str, unit: str | None) -> None:
        held = self.process(unit) if unit else ""
        self.unix.append(f"u_str LISTEN 0 4096 {path} 362417538 * 0 {held}")

    def audit(self, *args: str) -> subprocess.CompletedProcess[str]:
        (self.ss / "inet").write_text("".join(f"{line}\n" for line in self.inet))
        (self.ss / "unix").write_text("".join(f"{line}\n" for line in self.unix))
        return subprocess.run(
            [sys.executable, str(DEPLOY / "bin" / "audit-listeners"), *args],
            env=self.env,
            capture_output=True,
            text=True,
        )


def test_the_audit_passes_a_gpu_node_whose_servers_listen_on_their_sockets(tmp_path: Path) -> None:
    host = Listeners(tmp_path)
    lh = host.runtime / "lh"
    for unit, name in (("lh-engine", "engine"), ("lh-stt", "stt"), ("lh-tts", "tts")):
        host.socket(f"{lh}/{name}.sock", f"{unit}.service")
    host.socket(f"{lh}/8c3f0b2e-0d2c-4a51-9d55-0f6a2c0b7f3e", "lh-engine.service")  # vLLM's ZMQ IPC
    host.tcp("127.0.0.1:631", None)  # another user's
    host.socket("/run/systemd/private", None)
    sockets = [f"{lh}/{name}.sock" for name in ("engine", "stt", "tts")]
    done = host.audit(*(arg for path in sockets for arg in ("--expect", path)))
    assert done.returncode == 0, done.stderr
    assert done.stdout == (
        f"audit-listeners: ok: the box's units hold 4 Unix sockets, all in {host.runtime}, "
        "and 0 TCP or UDP listeners, all allowed\n"
    )


def test_the_audit_fails_on_any_listener_of_ours_outside_the_allowed_set(tmp_path: Path) -> None:
    host = Listeners(tmp_path)
    # Today's box: each server on a loopback port that every user of the host reaches.
    host.tcp("127.0.0.1:8000", "lh-engine.service")
    host.tcp("127.0.0.1:8100", "lh-stt.service")
    host.tcp("127.0.0.1:8880", "lh-tts.service")
    host.socket("/tmp/8c3f0b2e-0d2c-4a51-9d55-0f6a2c0b7f3e", "lh-engine.service")
    host.socket("@lh-abstract", "lh-tts.service")
    host.runtime.joinpath("lh").chmod(0o755)
    done = host.audit("--expect", f"{host.runtime}/lh/engine.sock")
    assert done.returncode == 1
    assert done.stderr.splitlines() == [
        f"audit-listeners: {host.runtime}/lh is mode 755, owned by uid {os.getuid()}: it must be 0700 and ours",
        "audit-listeners: lh-engine.service listens on tcp 127.0.0.1:8000",
        "audit-listeners: lh-stt.service listens on tcp 127.0.0.1:8100",
        "audit-listeners: lh-tts.service listens on tcp 127.0.0.1:8880",
        "audit-listeners: lh-engine.service listens on a socket outside "
        f"{host.runtime}: /tmp/8c3f0b2e-0d2c-4a51-9d55-0f6a2c0b7f3e",
        f"audit-listeners: lh-tts.service listens on an abstract socket outside {host.runtime}: @lh-abstract",
        f"audit-listeners: nothing listens on {host.runtime}/lh/engine.sock",
    ]


def test_the_audit_allows_the_gateways_port_and_an_sglang_engine_on_loopback(
    tmp_path: Path,
) -> None:
    host = Listeners(tmp_path)
    host.tcp("127.0.0.1:18080", "lh-gateway.service")
    host.tcp("127.0.0.1:8000", "lh-engine.service", "python -m sglang.launch_server")
    host.socket(f"{host.runtime}/lh/phastform-engine.sock", "lh-tunnel@phastform.service")
    assert host.audit().returncode == 0
    # Beyond loopback, and under any other unit, the same addresses fail.
    host.tcp("0.0.0.0:29500", "lh-engine.service", "python -m sglang.launch_server")
    host.tcp("127.0.0.1:18000", "lh-tunnel-phastform.service", "ssh -N")
    done = host.audit()
    assert done.returncode == 1
    assert done.stderr.splitlines() == [
        "audit-listeners: lh-engine.service listens on tcp 0.0.0.0:29500",
        "audit-listeners: lh-tunnel-phastform.service listens on tcp 127.0.0.1:18000",
    ]


def test_the_audit_fails_when_another_user_holds_the_gateways_port(tmp_path: Path) -> None:
    host = Listeners(tmp_path)
    host.tcp("127.0.0.1:18080", None)
    done = host.audit()
    assert done.returncode == 1
    assert done.stderr == (
        "audit-listeners: 127.0.0.1:18080 is held by a process that is not lh-gateway.service's: "
        "another user's?\n"
    )
