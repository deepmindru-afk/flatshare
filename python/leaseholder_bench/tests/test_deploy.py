"""The deployment: what the GPU nodes serve and the tunnels reach, cyan's install, and the script that
authorizes cyan's tunnel keys on a GPU node, run against stand-ins for ssh and systemd."""

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
KEYS = {
    "engine": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMa9oQ1uNw4Z5fT1n2x0bq7d1H0mGJ1s7mYcQjv8Yw5k",
    "stt": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIK3fW8pQ2nV7xZ1cR5tY9uB4mE6hJ0aLsD2gF8kN3qPw",
    "tts": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHq2wE4rT6yU8iO0pA1sD3fG5hJ7kL9zX2cV4bN6mQ8w",
}
"""cyan's tunnel keys, by the socket each one reaches: their type and blob, as ssh-keygen writes them before the
comment."""
FORMER = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPz4xY6wQ8eR0tU2iO4pA6sD8fG0hJ2kL4zX6cV8bN0mQ2e"
"""cyan's former tunnel key, which forwarded the node's ports."""
LOGIN = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIB7m2Qp9xR4tYc3vLk8wNs6fJd0aHe5uPq1zXo2iGb4M me@laptop"
"""Another key in a GPU node's authorized_keys: its user's own."""

FAKE_SSH = """\
#!/bin/sh
# ssh HOST COMMAND...: runs COMMAND in HOST's home under $FAKE_HOSTS, as sshd runs it in the user's shell.
host=$1
shift
cd "$FAKE_HOSTS/$host" && exec sh -c "$*"
"""
OPENBSD_NC = """\
#!/bin/sh
# nc -h on a GPU node: OpenBSD's netcat, as Ubuntu's netcat-openbsd prints its usage.
echo "OpenBSD netcat (Debian patchlevel 1.218-4ubuntu1)" >&2
exit 1
"""


def authorize(
    work: Path,
    keys: dict[str, str],
    authorized_keys: str = "",
    former: str | None = None,
    nc: str = OPENBSD_NC,
) -> tuple[subprocess.CompletedProcess[str], str]:
    """Runs authorize-tunnel.sh from cyan to phastform, hosts kept under ``work``: cyan's tunnel keys are
    ``keys`` by socket (and its former key ``former``), phastform's authorized_keys ``authorized_keys``
    and its nc ``nc``. Returns the run and the file it leaves."""
    hosts, bin_dir = work / "hosts", work / "bin"
    for host in ("cyan", "phastform"):
        (hosts / host / ".ssh").mkdir(parents=True)
    for name, key in keys.items():
        (hosts / "cyan" / ".ssh" / f"lh_tunnel_{name}_ed25519.pub").write_text(f"{key}\n")
    if former is not None:
        (hosts / "cyan" / ".ssh" / "lh_tunnel_ed25519.pub").write_text(f"{former}\n")
    authorized = hosts / "phastform" / ".ssh" / "authorized_keys"
    authorized.write_text(authorized_keys)
    bin_dir.mkdir()
    for tool, text in (("fake-ssh", FAKE_SSH), ("nc", nc)):
        (bin_dir / tool).write_text(text)
        (bin_dir / tool).chmod(0o755)
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


def test_the_gpu_nodes_publish_no_kv_events_and_cyan_reaches_them_only_through_the_tunnel() -> None:
    # vLLM's KV-event publisher connects to a TCP endpoint and serves a replay of the events, which carry
    # every cached block's token ids and session id: on a shared host any user could bind or ask for them,
    # and nothing of the deployment reads them.
    engine = (DEPLOY / "phastform" / "engine.yaml").read_text()
    assert "kv-events-config" not in engine and "tcp://" not in engine
    cyan = DEPLOY / "cyan" / "systemd"
    tunnel = (cyan / "lh-tunnel@.service").read_text()
    assert exec_lines(tunnel, "ExecStart") == [
        "/usr/bin/python3 %h/leaseholder/src/deploy/bin/lh-tunnel %i"
    ]
    assert "EnvironmentFile=%h/.config/leaseholder/tunnel-%i.env" in tunnel.splitlines()
    # One template for every GPU node: no unit forwards a port, or names a tunnel unit that is gone.
    for unit in cyan.iterdir():
        text = unit.read_text()
        assert "ssh " not in text and "lh-tunnel-" not in text, unit.name


def test_each_tunnel_key_may_only_bridge_its_session_to_its_socket(tmp_path: Path) -> None:
    lh = f"/run/user/{os.getuid()}/lh"
    earlier = (
        f'restrict,port-forwarding,permitopen="127.0.0.1:5558" {KEYS["stt"]} lh-tunnel-stt@cyan'
    )
    former = f'restrict,port-forwarding,permitopen="127.0.0.1:8000",command="/bin/false" {FORMER} lh-tunnel@cyan'
    # The keys as ssh-keygen writes them, and without their comments, which are optional.
    for case, comment in (("commented", " lh-tunnel-{}@cyan"), ("bare", "")):
        keys = {name: key + comment.format(name) for name, key in KEYS.items()}
        done, authorized = authorize(
            tmp_path / case, keys, f"{LOGIN}\n{earlier}\n{former}\n", former=FORMER
        )
        assert done.returncode == 0, done.stderr
        # The earlier entry for a key and the former key's go; every other key stays. Each entry allows its
        # key's sessions one command, the bridge to the node's socket of its name, and no forwarding.
        assert authorized.splitlines() == [LOGIN] + [
            f'restrict,command="exec nc -q0 -U {lh}/{name}.sock" {keys[name]}' for name in KEYS
        ], case
        assert done.stdout == f"authorized: engine, stt and tts on {lh} (3 entries)\n", case


def test_authorizing_needs_all_three_keys_and_openbsd_netcat(tmp_path: Path) -> None:
    # A client without one of the keys changes nothing.
    done, authorized = authorize(tmp_path / "missing", {"engine": KEYS["engine"]}, f"{LOGIN}\n")
    assert (
        done.returncode == 2
        and "no public key in cyan's .ssh/lh_tunnel_stt_ed25519.pub" in done.stderr
    )
    assert authorized == f"{LOGIN}\n"
    # Nor does a node whose nc cannot bridge a session to a socket.
    gnu = "#!/bin/sh\necho 'GNU netcat 0.7.1, a rewrite of the famous networking tool.'\n"
    done, authorized = authorize(tmp_path / "gnu", KEYS, f"{LOGIN}\n", nc=gnu)
    assert done.returncode == 2 and "is not OpenBSD netcat" in done.stderr
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


def exec_lines(text: str, key: str) -> list[str]:
    """The values of a unit's ``key=`` lines, continued lines joined."""
    lines = text.replace("\\\n", " ").splitlines()
    return [line.partition("=")[2] for line in lines if line.startswith(f"{key}=")]


def test_the_gpu_nodes_servers_listen_on_their_sockets_in_a_private_directory() -> None:
    phastform = DEPLOY / "phastform"
    for config in ("engine.yaml", "stt.yaml"):
        keys = re.findall(r"^([a-z-]+):", (phastform / config).read_text(), re.M)
        assert "host" not in keys and "port" not in keys, config
    sockets = {
        "lh-engine": "${LH_ENGINE_SOCKET}",
        "lh-stt": "${LH_STT_SOCKET}",
        "lh-tts": "%t/lh/tts.sock",
    }
    for unit, socket in sockets.items():
        text = (phastform / "systemd" / f"{unit}.service").read_text()
        [start] = exec_lines(text, "ExecStart")
        assert f"--uds {socket}" in start and "--port" not in start and "--host" not in start, unit
        pre = exec_lines(text, "ExecStartPre")
        assert pre[0] == "/usr/bin/install -d -m 0700 %t/lh", unit
        if (
            unit != "lh-tts"
        ):  # vLLM leaves a killed server's socket file; uvicorn replaces it itself
            assert f"/usr/bin/rm -f {socket}" in pre, unit
        [ready] = exec_lines(text, "ExecStartPost")
        assert ready.startswith(
            f"%h/leaseholder/src/deploy/bin/wait-healthy --unix-socket {socket} "
        ), unit
    engine = (phastform / "systemd" / "lh-engine.service").read_text()
    assert "Environment=LH_ENGINE_SOCKET=%t/lh/engine.sock" in engine.splitlines()
    # Every variant that starts its own vLLM serves the same socket; SGLang, which cannot, gets a proxy to its port.
    for conf in (phastform / "variants").glob("*/lh-engine.conf"):
        starts = exec_lines(conf.read_text(), "ExecStart")
        if not starts:
            continue
        if "sglang" in starts[-1]:
            assert "Wants=lh-engine-proxy.socket" in conf.read_text().splitlines(), conf
        else:
            assert "--uds ${LH_ENGINE_SOCKET} $LH_ENGINE_FLAGS" in starts[-1], conf
    proxy = (phastform / "systemd" / "lh-engine-proxy.socket").read_text().splitlines()
    assert "ListenStream=%t/lh/engine.sock" in proxy and "BindsTo=lh-engine.service" in proxy
    smoke = (phastform / "smoke.sh").read_text()
    for name in ("engine", "stt", "tts"):
        assert f"${{lh}}/{name}.sock" in smoke or f"$lh/{name}.sock" in smoke, name
    assert '"$(dirname "$0")/../bin/audit-listeners"' in smoke
