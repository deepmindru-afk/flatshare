"""rehearse.sh on a CPU: the single-host rehearsal's every step against stand-ins for the hosts (directories that a fake
ssh runs commands in, with the real rsync between them), the box's user manager, nvidia-smi and ss, and a stub of the
kit's bootstrap.sh, run.sh and lh-svc shims; deploy/a100/push.sh and pull.sh are the repository's own. What it holds:
the checks refuse a box that runs, a busy GPU or a held port; nothing of the rehearsal reaches the box's deployment
(its source, its units, its user manager), and cleanup finds it as it was, or says what changed; the environment the
kit runs in is the rehearsal's (its home, runtime directory, lh-svc, the box's hub cache offline, no sudo); a step on
the host is detached and waited for, and a step done is not run again but under --redo; block 0's listener audit must
pass with each GPU server in a network namespace of its own (deploy/bin/in-private-net); block 4 is run for one point and
its deadline's stop; pull copies the rehearsal's records, not the box's; reproduce runs make on peacock over them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
HERE = REPO / "benchmarks" / "a100" / "rehearsal"
FAKES = REPO / "python" / "leaseholder_bench" / "tests" / "fakes"
REHEARSE = HERE / "rehearse.sh"
HOST_SH = HERE / "host.sh"
QWEN, QWEN_REV = "Qwen/Qwen3-4B-Instruct-2507-FP8", "8591804019c8b22094c3b5b4454e0edc05dffc98"
WHISPER, WHISPER_REV = "openai/whisper-large-v3-turbo", "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"

TOOLS = {
    # The box's user manager: it answers through the login session's bus, and not where no bus is named (the
    # rehearsal's environment). Every call is logged: the test holds that none acts on a unit.
    "systemctl": """\
#!/usr/bin/env bash
if [[ -z ${DBUS_SESSION_BUS_ADDRESS:-} ]]; then echo "Failed to connect to bus: No such file or directory" >&2; exit 1; fi
echo "systemctl $*" >>"$FAKE_BOX/.systemctl.log"
case "$*" in
  *is-system-running*) echo running ;;
  *list-unit-files*) cat "$FAKE_BOX/.unit-files" ;;
  *list-units*) cat "$FAKE_BOX/.units" ;;
esac
""",
    "nvidia-smi": """\
#!/usr/bin/env bash
case "$*" in
  *--query-compute-apps*) cat "$FAKE_BOX/.gpu-apps" 2>/dev/null ;;
  *) echo "NVIDIA GeForce RTX 5060 Ti, 2 MiB, 16311 MiB" ;;
esac
""",
    "ss": """\
#!/usr/bin/env bash
case "$*" in
  *-Hltn*) cat "$FAKE_BOX/.listen-tcp" 2>/dev/null ;;
  *-Hlun*) cat "$FAKE_BOX/.listen-udp" 2>/dev/null ;;
esac
""",
    "unshare": "#!/bin/sh\nexit 0\n",
    # make on peacock: reproduce-a100 over the records it is given, as analysis.py writes its sections.
    "make": """\
#!/usr/bin/env bash
for a in "$@"; do case $a in RECORDS=*) records=${a#RECORDS=} ;; CORPUS=*) corpus=${a#CORPUS=} ;; esac; done
[[ " $* " == *" reproduce-a100 "* && -f $records/runs/a100/blocks.log && -d $corpus ]] || { echo "make: $*" >&2; exit 2; }
echo "make $*" >>"$HOME/make.log"
mkdir -p build/reproduce-a100
printf '%s\\n' "# The A100 40 GB run against its forecast" "" "## 1. The step model" "" "Not measured: no calibration point." \\
  "" "## 3. The probes" "" "- frozen: KV case 1650 ms" "" "## 4. E2 within the A100" "" "| point | v2v |" \\
  >build/reproduce-a100/a100.md
""",
}

STUB_BOOTSTRAP = """\
#!/usr/bin/env bash
# bootstrap.sh in the rehearsal's test: the environment it ran in, and what each stage leaves for rehearse.sh's checks.
set -euo pipefail
stage=$1
src=$(cd "$(dirname "$0")/../.." && pwd)
root=$HOME/leaseholder
node=$root/node
mkdir -p "$node" "$root/cache"
{
  env | grep -E '^(HOME|XDG_RUNTIME_DIR|XDG_STATE_HOME|LH_SUPERVISOR|HF_HUB_CACHE|HF_HUB_OFFLINE|DBUS_SESSION_BUS_ADDRESS|TMPDIR|LH_REHEARSAL)=' | sort
  if sudo -n true 2>/dev/null; then echo "sudo worked"; else echo "sudo refused"; fi
} >"$root/bootstrap-$stage.env"
echo "bootstrap $stage: from $src"
[[ ${FAKE_FAIL:-} != "$stage" ]] || { echo "bootstrap $stage: failing, as the test asks"; exit 1; }
if [[ $stage == stage ]]; then
  printf '%s\\n' 'export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-'"$HOME"'/.local/state/lh-svc/run}' \\
    "export PATH=$src/deploy/bin/shims:\\$PATH" 'export LH_SUPERVISOR=lh-svc' "export HF_HUB_CACHE=$HF_HUB_CACHE" >"$node/env.sh"
  source "$node/env.sh"
  systemctl --user start lh-voice-worker.service
fi
[[ $stage != warm ]] || echo '{"configurations": {"frozen": {"ok": true}, "pr": {"ok": true}}}' >"$root/cache/warm.json"
echo '{"counts": {"ok": 20, "degraded": 2, "refused": 0, "pending": 0}, "checks": [{"check": "Cores", "status": "degraded", "value": "16 CPUs"}]}' >"$node/preflight.json"
echo "bootstrap $stage: done (no check refused)"
"""

STUB_RUN = """\
#!/usr/bin/env bash
# run.sh in the rehearsal's test (it takes --until, --cap and --estimate BLOCK=MINUTES as the kit's, and skips a block
# done): the blocks' records, and block 0's listener audit as FAKE_AUDIT says.
set -euo pipefail
root=$HOME/leaseholder
runs=$root/runs/a100
mkdir -p "$runs/blocks"
log() { echo "[$(date -u +%FT%TZ)] run: $*" | tee -a "$runs/blocks.log"; }
args="$*"
until="" estimate=""
while [[ ${1:-} == --* ]]; do
  case $1 in --until) until=$2 ;; --estimate) estimate=$2 ;; --cap) ;; esac
  shift 2
done
log "blocks $*${until:+, until $until UTC}"
for block in "$@"; do
  if [[ -f $runs/blocks/$block.done ]]; then
    log "block $block: done ($(cat "$runs/blocks/$block.done"))"
    continue
  fi
  case $block in
    0)
      mkdir -p "$runs/block0"
      printf '%-26s %s\\n' health ok "tts question (wav)" "0.3s" stt "0.4s: What is the capital of France?" llm "0.1s: Paris." \\
        "tts reply (first/total)" "0.2s/0.4s" "gpu memory" "13900 MiB, 16311 MiB" >"$runs/block0/smoke.txt"
      own="lh-engine, lh-stt, lh-tts in network namespaces of their own, with 12 TCP or UDP listeners and 3 abstract sockets there"
      case ${FAKE_AUDIT:-private} in
        private) audit="ok: the box's units hold 5 Unix sockets, all in $XDG_RUNTIME_DIR, and 1 TCP or UDP listeners, all allowed; $own" ;;
        degraded) audit="ok: the box's units hold 5 Unix sockets, all in $XDG_RUNTIME_DIR, and 1 TCP or UDP listeners, all allowed; the GPU servers run in the host's network namespace (LH_PRIVATE_NET=host) with 12 loopback TCP listeners and 3 abstract sockets" ;;
        refused)
          echo "audit-listeners: lh-engine.service listens on tcp 127.0.1.1:41234" >&2
          echo "smoke: the listener audit failed (above)" >&2
          exit 1 ;;
      esac
      printf '%-26s %s\\n' "listener audit" "$audit" >>"$runs/block0/smoke.txt"
      echo "smoke: ok" >>"$runs/block0/smoke.txt"
      echo "bots: 1 caller, 2 turns, 0 failed" >"$runs/block0/call.txt" ;;
    1)
      mkdir -p "$runs/block1/frozen-e0"
      echo '{}' >"$runs/block1/frozen-e0/summary.json"
      for f in kv-1 kv-2 kv-3 prefill-1 prefill-2 prefill-3 tail-1 tail-2; do echo "$f: priority-0 TTFT 1650 ms" >"$runs/block1/frozen-$f.txt"; done ;;
    4)
      [[ -n $until && $estimate == 4=15 ]] || { echo "run.sh: the rehearsal's block 4 wants --until and --estimate 4=15" >&2; exit 3; }
      echo "$args" >"$runs/block4.args"
      points=$root/results/e2/a100-r1
      mkdir -p "$points/w5-disc-pr-ab48-s24"
      echo '{"status": "complete"}' >"$points/w5-disc-pr-ab48-s24/run.json"
      echo "[t] w5-a1-voice: 9 minutes would end after $until UTC; stopping before it (--until)" >>"$points/campaign.log"
      log "a100-r1 stopped before its deadline ($until UTC) with points left"
      exit 0 ;;
  esac
  date -u +%FT%TZ >"$runs/blocks/$block.done"
  log "block $block: done"
done
"""

STUB_SYSTEMCTL = """\
#!/usr/bin/env bash
# deploy/bin/shims/systemctl in the rehearsal's test: lh-svc's units, their states in the rehearsal's state directory.
dir=${XDG_STATE_HOME:?}/fake-lh-svc
mkdir -p "$dir"
echo "systemctl $*" >>"$dir/calls"
words=()
for a in "$@"; do [[ $a == --* ]] || words+=("$a"); done
action=${words[0]}
for unit in "${words[@]:1}"; do
  unit=${unit%.service}
  case $action in
    start)
      echo active >"$dir/$unit"
      if [[ $unit == lh-voice-worker ]]; then for u in lh-voicenet lh-livekit lh-gateway; do echo active >"$dir/$u"; done; fi ;;
    stop) echo inactive >"$dir/$unit" ;;
    is-active) cat "$dir/$unit" 2>/dev/null || echo inactive ;;
  esac
done
"""

STUB_JOURNALCTL = '#!/usr/bin/env bash\necho "1761400000.000000 phastform lh-engine[42]: a line of the engine\'s"\n'


def write(path: Path, text: str, mode: int = 0o644) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)
    return path


@pytest.fixture
def world(tmp_path: Path) -> dict:
    """The workstation's checkout and data, the box (phastform's login home with its deployment), cyan and peacock."""
    hosts = tmp_path / "hosts"
    box = hosts / "phastform"
    # The box's deployment, as the rehearsal must leave it.
    write(box / "leaseholder" / "src" / "DEPLOYED", "v0.3-191-g987fd09\n")
    write(
        box / "leaseholder" / "src" / "deploy" / "phastform" / "engine.yaml", "model: the box's\n"
    )
    write(box / "leaseholder" / "runs" / "box-only.txt", "a record of the box's\n")
    write(box / "leaseholder" / "venvs" / "engine" / "bin" / "python", "#!/bin/sh\n", 0o755)
    kokoro = box / "leaseholder" / "vendor" / "Kokoro-FastAPI"
    write(kokoro / ".git" / "HEAD", "ebd4ec9\n")
    write(kokoro / "api" / "src" / "models" / "v1_0" / "kokoro-v1_0.pth", "weights\n")
    (box / ".config" / "systemd" / "user").mkdir(parents=True)
    (box / ".config" / "systemd" / "user" / "lh-engine.service").symlink_to(
        box / "leaseholder" / "src" / "deploy" / "phastform" / "engine.yaml"
    )
    hub = box / ".cache" / "huggingface" / "hub"
    for model, revision in ((QWEN, QWEN_REV), (WHISPER, WHISPER_REV)):
        write(
            hub / f"models--{model.replace('/', '--')}" / "snapshots" / revision / "config.json",
            "{}\n",
        )
    write(box / ".units", "lh-engine.service loaded inactive dead Leaseholder LLM engine\n")
    write(
        box / ".unit-files",
        "lh-engine.service linked disabled\nleaseholder.target linked enabled\n",
    )
    (box / ".gpu-apps").write_text("")
    write(box / ".listen-tcp", "LISTEN 0 128 0.0.0.0:22 0.0.0.0:*\n")
    write(box / ".listen-udp", "")
    write(
        hosts / "cyan" / "leaseholder" / "data" / "callers" / "manifest.json",
        json.dumps({"hello": {"kind": "turn"}}),
    )
    write(hosts / "cyan" / "leaseholder" / "data" / "callers" / "hello.wav", "RIFF")
    write(hosts / "peacock" / "leaseholder-data" / "weka-x32" / "t.json", "{}")
    # The workstation's corpus, 50 traces, and the checkout to push: the kit's push.sh and pull.sh, and stubs.
    corpus = tmp_path / "corpus" / "weka-x32"
    corpus.mkdir(parents=True)
    for n in range(50):
        (corpus / f"trace-{n:02d}.json").write_text(
            json.dumps({"requests": [{"in": 100 * (n + 1)}]})
        )
    repo = tmp_path / "checkout"
    for rel in (
        "deploy/a100/push.sh",
        "deploy/a100/pull.sh",
        "deploy/phastform/engine.yaml",
        "deploy/phastform/stt.yaml",
        "benchmarks/e2/long_traces.py",
        "deploy/bin/in-private-net",
    ):
        write(repo / rel, (REPO / rel).read_text(), 0o755)
    write(repo / "deploy" / "a100" / "bootstrap.sh", STUB_BOOTSTRAP, 0o755)
    write(repo / "benchmarks" / "a100" / "run.sh", STUB_RUN, 0o755)
    write(repo / "deploy" / "bin" / "shims" / "systemctl", STUB_SYSTEMCTL, 0o755)
    write(repo / "deploy" / "bin" / "shims" / "journalctl", STUB_JOURNALCTL, 0o755)
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "the kit"], check=True)
    bin_dir = tmp_path / "bin"
    for name, text in TOOLS.items():
        write(bin_dir / name, text, 0o755)
    # A host whose kernel refuses unprivileged user namespaces.
    write(tmp_path / "no-userns" / "unshare", "#!/bin/sh\nexit 1\n", 0o755)
    (bin_dir / "ssh").symlink_to(FAKES / "campaign" / "ssh")
    work = tmp_path / "work"
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("REHEARSAL_", "LH_", "XDG_", "HF_")) and k != "VIRTUAL_ENV"
    } | {
        "PATH": os.pathsep.join([str(bin_dir), os.environ["PATH"]]),
        "FAKE_HOSTS": str(hosts),
        "FAKE_BOX": str(box),
        # The login environment's session bus, through which the box's user manager answers.
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path={box}/bus",
        "REHEARSAL_HOST": "phastform",
        "REHEARSAL_SSH": "ssh",
        "REHEARSAL_CYAN": "cyan",
        "REHEARSAL_PEACOCK": "peacock",
        "REHEARSAL_REPO": str(repo),
        "REHEARSAL_WORK": str(work),
        "REHEARSAL_CORPUS": str(corpus),
        "REHEARSAL_MIN_FREE_GIB": "1",
        "REHEARSAL_POLL_S": "0.2",
    }
    return {
        "tmp": tmp_path,
        "hosts": hosts,
        "box": box,
        "hub": hub,
        "repo": repo,
        "env": env,
        "work": work,
        "state": work / "state",
    }


def rehearse(world: dict, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(REHEARSE), *args],
        env=world["env"] | env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def box_snapshot(box: Path) -> dict[str, str]:
    """The box's deployment: each file's contents and each link's target, under the trees the rehearsal must leave."""
    found = {}
    for top in ("leaseholder", ".config", ".cache"):
        for path in sorted((box / top).rglob("*")):
            if path.is_symlink():
                found[str(path.relative_to(box))] = "-> " + os.readlink(path)
            elif path.is_file():
                found[str(path.relative_to(box))] = path.read_text(errors="replace")
    return found


def test_the_checks_refuse_a_box_that_runs_a_busy_gpu_and_a_held_port(world: dict) -> None:
    box = world["box"]
    write(box / ".units", "lh-engine.service loaded active running Leaseholder LLM engine\n")
    write(box / ".gpu-apps", "4242, VLLM::EngineCore, 13000 MiB\n")
    write(box / ".listen-tcp", "LISTEN 0 128 127.0.0.1:18080 0.0.0.0:*\n")
    write(box / ".listen-udp", "UNCONN 0 0 0.0.0.0:17882 0.0.0.0:*\n")
    done = rehearse(world, "check")
    assert done.returncode == 1, done.stdout + done.stderr
    out = done.stdout
    assert "check: the box's units: refused: running: lh-engine.service" in out
    assert "check: GPU: refused: in use: 4242, VLLM::EngineCore" in out
    assert "check: ports: refused: held: tcp 127.0.0.1:18080 udp 0.0.0.0:17882" in out
    assert not (world["state"] / "check.done").exists()
    # Each cleared, the host is ready; nothing was written on it.
    write(box / ".units", "lh-engine.service loaded inactive dead Leaseholder LLM engine\n")
    (box / ".gpu-apps").write_text("")
    write(box / ".listen-tcp", "LISTEN 0 128 0.0.0.0:22 0.0.0.0:*\n")
    (box / ".listen-udp").write_text("")
    done = rehearse(world, "check")
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"check: hub cache: ok: {world['hub']}" in done.stdout
    assert f"check: model {QWEN}: ok: {QWEN_REV}" in done.stdout
    assert not (box / "a100-rehearsal").exists()


def test_the_rehearsal_runs_every_step_in_its_own_home_and_leaves_the_box_as_it_was(
    world: dict,
) -> None:
    box, hosts, state = world["box"], world["hosts"], world["state"]
    before = box_snapshot(box)
    for step in ("check", "init", "data", "push", "stage", "gpu", "warm"):
        done = rehearse(world, step)
        assert done.returncode == 0, (step, done.stdout + done.stderr)
    home = box / "a100-rehearsal"
    root = home / "leaseholder"
    # The rehearsal's environment: its own home, runtime directory and state, lh-svc, the box's hub cache offline, no
    # session bus, sudo refused; and the box's user manager out of its reach.
    recorded = (root / "bootstrap-stage.env").read_text().splitlines()
    assert f"HOME={home}" in recorded and f"XDG_RUNTIME_DIR={home}/run" in recorded
    assert f"HF_HUB_CACHE={world['hub']}" in recorded and "HF_HUB_OFFLINE=1" in recorded
    assert "LH_SUPERVISOR=lh-svc" in recorded and "sudo refused" in recorded
    assert not any(line.startswith("DBUS_SESSION_BUS_ADDRESS=") for line in recorded)
    assert (
        "the box's user manager is out of the rehearsal's reach"
        in (state / "logs" / "init.log").read_text()
    )
    assert (home / "run").stat().st_mode & 0o777 == 0o700
    # The box's Kokoro, copied; the data pushed as files; the source at the checkout's revision, the box's untouched.
    assert (
        root / "vendor" / "Kokoro-FastAPI" / "api" / "src" / "models" / "v1_0" / "kokoro-v1_0.pth"
    ).exists()
    long = sorted((root / "data" / "weka-x32-long").iterdir())
    assert len(long) == 48 and not any(p.is_symlink() for p in long)
    assert (root / "data" / "callers" / "manifest.json").exists()
    revision = subprocess.run(
        ["git", "-C", str(world["repo"]), "describe", "--always", "--dirty"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert (root / "src" / "DEPLOYED").read_text().strip() == revision
    assert box_snapshot(box) == before
    # The control plane up under the rehearsal's lh-svc; the box's user manager was asked only for what it runs.
    calls = (box / ".systemctl.log").read_text().splitlines()
    assert calls and all("list-unit" in c for c in calls), calls
    assert (
        "start lh-voice-worker.service"
        in (home / ".local" / "state" / "fake-lh-svc" / "calls").read_text()
    )
    # A step done is not run again; its host step is done.
    done = rehearse(world, "stage")
    assert done.returncode == 0 and "stage: done before" in done.stdout, done.stdout + done.stderr
    assert "bootstrap stage" not in done.stdout
    # Block 0: a listener audit that refuses stops the rehearsal; one that passes with the GPU servers in the host's
    # namespace, where this host allows user namespaces, fails block 0's check; a step done but not checked runs
    # again only under --redo (the block's blocks/0.done removed); with each GPU server in a namespace of its own
    # (deploy/bin/in-private-net), block 0 is done.
    done = rehearse(world, "block0", FAKE_AUDIT="refused")
    assert done.returncode == 1 and "block0: run.sh 0 failed" in done.stdout, (
        done.stdout + done.stderr
    )
    assert not (state / "block0.done").exists()
    done = rehearse(world, "block0", FAKE_AUDIT="degraded")
    assert done.returncode == 1, done.stdout + done.stderr
    assert (
        "the listener audit's line is not ok with lh-engine, lh-stt, lh-tts in network namespaces of their own"
        in done.stdout
    )
    done = rehearse(world, "block0")
    assert done.returncode == 1 and "block0 is done" in done.stdout, done.stdout
    done = rehearse(world, "--redo", "block0", "block0")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "in network namespaces of their own" in done.stdout
    # Where the kernel refuses user namespaces (the A100's Ubuntu 24.04), the degraded line is the one expected.
    done = subprocess.run(
        ["bash", str(HOST_SH), "verify", "block0"],
        cwd=home,
        env=world["env"]
        | {"HOME": str(home), "PATH": f"{world['tmp'] / 'no-userns'}:{world['env']['PATH']}"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 1 and "LH_PRIVATE_NET=host" in done.stdout, done.stdout
    # Block 1, then block 4 for its first point and the deadline's stop, its deadline 16 to 17 minutes on.
    for step in ("block1", "block4"):
        done = rehearse(world, step)
        assert done.returncode == 0, (step, done.stdout + done.stderr)
    args = (root / "runs" / "a100" / "block4.args").read_text().split()
    assert args[0] == "--until" and args[2:] == ["--estimate", "4=15", "4"], args
    hours, minutes = map(int, args[1].split(":"))
    now = time.gmtime()
    ahead = (hours * 60 + minutes - now.tm_hour * 60 - now.tm_min) % 1440
    assert 16 <= ahead <= 17, (args[1], ahead)
    assert "the first point complete, then: w5-a1-voice: 9 minutes would end after" in done.stdout
    # pull copies the rehearsal's records, not the box's; reproduce runs make on peacock over them.
    done = rehearse(world, "pull")
    assert done.returncode == 0, done.stdout + done.stderr
    pulled = world["work"] / "hosts" / "phastform"
    assert (pulled / "runs" / "a100" / "blocks.log").exists()
    assert not (pulled / "runs" / "box-only.txt").exists()
    assert "a line of the engine's" in (pulled / "runs" / "journal" / "units.log").read_text()
    done = rehearse(world, "reproduce")
    assert done.returncode == 0, done.stdout + done.stderr
    assert (world["work"] / "reproduce" / "a100.md").read_text().startswith("# The A100 40 GB run")
    assert "make reproduce-a100 RECORDS=../records" in (hosts / "peacock" / "make.log").read_text()
    # cleanup: the rehearsal's units stopped, the box as it was; one change to the box is found and named.
    done = rehearse(world, "cleanup")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "verify: the box's deployment as the rehearsal found it" in done.stdout
    stops = [
        line
        for line in (home / ".local" / "state" / "fake-lh-svc" / "calls").read_text().splitlines()
        if line.startswith("systemctl --user stop ")
    ]
    assert stops and all(
        unit in stops[-1].split()
        for unit in (
            "leaseholder.target",
            "lh-engine.service",
            "lh-voice-worker.service",
            "lh-gateway.socket",
        )
    ), stops
    assert box_snapshot(box) == before
    (box / "leaseholder" / "src" / "DEPLOYED").write_text("changed\n")
    done = rehearse(world, "--redo", "cleanup", "cleanup")
    assert done.returncode == 1, done.stdout + done.stderr
    assert (
        "verify: the box's deployment changed" in done.stdout
        and "leaseholder/src/DEPLOYED" in done.stdout
    )
    status = rehearse(world, "status")
    assert status.returncode == 0 and "block4     done" in status.stdout, status.stdout


def test_a_step_on_the_host_outlives_its_session_and_is_waited_for_once(tmp_path: Path) -> None:
    # host.sh's detached steps, on this host: a step started runs on after its starter returns, a second start while
    # it runs starts nothing, wait returns its exit status, and stop ends one that runs.
    home = tmp_path / "home"
    home.mkdir()
    env = os.environ | {"HOME": str(home), "REHEARSAL_POLL_S": "0.1"}

    def host_sh(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(HOST_SH), *args], env=env, capture_output=True, text=True, timeout=60
        )

    started = host_sh("start", "slow", "sleep 1; echo slept; exit 3")
    assert started.returncode == 0 and "slow started (running" in started.stdout, started.stdout
    assert host_sh("state", "slow").stdout.startswith("running ")
    again = host_sh("start", "slow", "echo twice")
    assert "runs already" in again.stdout
    waited = host_sh("wait", "slow")
    assert waited.returncode == 3 and "slept" in waited.stdout, waited.stdout
    assert host_sh("state", "slow").stdout.strip() == "done 3"
    assert "twice" not in (home / ".rehearsal" / "steps" / "slow.log").read_text()
    host_sh("start", "long", "sleep 30")
    stopped = host_sh("stop", "long")
    assert (
        "long stopped" in stopped.stdout and host_sh("state", "long").stdout.strip() == "done 143"
    )
    done = host_sh("start", "ok", "true")
    assert done.returncode == 0
    assert host_sh("wait", "ok").returncode == 0
    assert "ok is done" in host_sh("start", "ok", "false").stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


@pytest.mark.skipif(not Path("/proc/self/environ").exists(), reason="reads /proc (Linux)")
def test_a_running_step_reads_as_running_whatever_its_environment_holds(tmp_path: Path) -> None:
    # host.sh tells a step's process by LH_REHEARSAL_STEP in its /proc/PID/environ. Read through tr and grep -q under
    # pipefail, tr died of SIGPIPE once grep had matched while more of the environment was to come, and the running
    # step read as died: here deterministically, the environment after the mark being larger than a pipe's buffer.
    steps = tmp_path / ".rehearsal" / "steps"
    steps.mkdir(parents=True)
    rest = {k: v for k, v in os.environ.items() if k != "LH_REHEARSAL_STEP"}
    env = {
        "LH_REHEARSAL_STEP": "stage",
        **rest,
        "LH_BULK_A": "y" * 100_000,
        "LH_BULK_B": "y" * 100_000,
    }
    step = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], env=env)
    try:
        (steps / "stage.pid").write_text(f"{step.pid}\n")
        done = subprocess.run(
            ["bash", str(HOST_SH), "state", "stage"],
            env=os.environ | {"HOME": str(tmp_path)},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert done.stdout.strip() == f"running {step.pid}", done.stdout + done.stderr
    finally:
        step.kill()
        step.wait()
