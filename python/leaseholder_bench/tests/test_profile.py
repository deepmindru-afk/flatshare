"""deploy/profile.py: the frozen box's profile renders deploy/phastform byte for byte, the A100's derives its
values from a host's facts by the rules, the core plan shares a host's CPUs, and detect reads a host through
stand-ins for nvidia-smi, systemd and the kernel's files."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
DEPLOY = REPO / "deploy"
GIB = 1 << 30

spec = importlib.util.spec_from_file_location("lh_profile", DEPLOY / "profile.py")
profile = importlib.util.module_from_spec(spec)
sys.modules["lh_profile"] = profile
spec.loader.exec_module(profile)


def host(cpus: dict[str, list[int]] | None = None, **facts) -> dict:
    """An A100 host's facts: one NUMA node of 32 CPUs, 200 GiB of RAM, 100 GiB of /dev/shm, no CUDA toolkit."""
    base = {
        "hostname": "a100-vm",
        "gpu": {
            "name": "NVIDIA A100-SXM4-40GB",
            "memory_mib": 40960,
            "compute_cap": "8.0",
            "numa_node": None,
        },
        "cpus": cpus or {"0": list(range(32))},
        "memory": {"total_bytes": 200 * GIB, "available_bytes": 190 * GIB, "nodes": {}},
        "shm": {"path": "/dev/shm", "total_bytes": 100 * GIB, "free_bytes": 100 * GIB},
        "cuda_toolkit": None,
    }
    return base | facts


def render(tmp_path: Path, name: str, facts: dict | None) -> tuple[Path, dict]:
    out = tmp_path / name
    node = profile.render(profile.load(name), facts, out)
    return out, node


def test_the_box_profile_renders_todays_files(tmp_path: Path) -> None:
    assert profile.check(profile.load("rtx-5060-ti"), DEPLOY / "phastform") == []
    done = subprocess.run(
        [sys.executable, str(DEPLOY / "profile.py"), "check", "rtx-5060-ti"],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert "byte for byte" in done.stdout
    # Every rendered file is the box's, and the box's own scripts come along as they are.
    out, node = render(tmp_path, "rtx-5060-ti", None)
    for rel in profile.RENDERED + profile.COPIED:
        assert (out / rel).read_bytes() == (DEPLOY / "phastform" / rel).read_bytes(), rel
    assert node["plan"] is None and node["warnings"] == []


def test_a_value_that_differs_from_the_box_shows(tmp_path: Path) -> None:
    box = profile.load("rtx-5060-ti")
    box["engine"]["kv_cache_memory"] = 6 << 30
    box["text"]["audio_numa"] = "# NUMA node 0 for the audio servers."
    assert profile.check(box, DEPLOY / "phastform") == [
        "engine.yaml",
        "systemd/lh-stt.service",
        "systemd/lh-tts.service",
    ]
    done = subprocess.run(
        [
            sys.executable,
            str(DEPLOY / "profile.py"),
            "check",
            "rtx-5060-ti",
            "--against",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode != 0


def test_the_a100_profile_derives_its_values_from_the_host(tmp_path: Path) -> None:
    out, node = render(tmp_path, "a100-40gb", host())
    values, plan = node["values"], node["plan"]
    # R2: the box's 10.83 and 2.32 GiB budgets of the card's 40 GiB.
    assert (values["engine.gpu_memory_utilization"], values["stt.gpu_memory_utilization"]) == (
        "0.28",
        "0.06",
    )
    # R3: four times the box's 5 GiB pool, which /dev/shm and the RAM hold.
    assert values["engine.cpu_bytes_to_use"] == str(20 * GIB)
    engine = (out / "engine.yaml").read_text()
    assert "kv-cache-memory: 5368709120\n" in engine and "gpu-memory-utilization: 0.28\n" in engine
    assert "cpu_bytes_to_use: 21474836480\n" in engine
    assert (
        "max-num-batched-tokens: 512\n" in engine
        and "long-prefill-token-threshold: 256\n" in engine
    )
    assert "model: Qwen/Qwen3-4B-Instruct-2507-FP8\n" in engine
    assert "{{" not in engine and ": auto\n" not in engine
    # No CUDA toolkit on the host: the engine and STT use the venv's, as Gray does.
    unit = (out / "systemd" / "lh-engine.service").read_text()
    assert f"Environment=CUDA_HOME={profile.VENV_CUDA}\n" in unit
    assert (
        "--config %h/leaseholder/node/engine.yaml --uds ${LH_ENGINE_SOCKET} $LH_ENGINE_FLAGS"
        in unit
    )
    assert (
        f"Environment=CUDA_HOME={profile.VENV_CUDA}\n"
        in (out / "systemd" / "lh-stt.service").read_text()
    )
    make_room = (out / "variants" / "make-room" / "lh-engine.conf").read_text()
    assert "--config %h/leaseholder/node/engine.yaml --scheduler-cls" in make_room
    # R7 on 32 CPUs, under the 34 that give each component its want.
    assert plan == {
        "engine": "0-7",
        "audio": "8-11",
        "gateway": "12-13",
        "voice": "14-20",
        "aiperf": "21-27",
        "callers": "28-31",
        "engine_numa": "0",
        "audio_numa": "0",
    }
    assert f"CPUAffinity={plan['engine']}\n" in unit
    assert node["warnings"] == []
    assert json.loads((out / "node.json").read_text()) == node


def test_a_host_with_its_own_cuda_13_toolkit_compiles_with_it(tmp_path: Path) -> None:
    facts = host(cuda_toolkit={"version": "13.0", "path": "/usr/local/cuda-13.0"})
    out, _ = render(tmp_path, "a100-40gb", facts)
    unit = (out / "systemd" / "lh-engine.service").read_text()
    assert "Environment=CUDA_HOME=/usr/local/cuda-13.0\n" in unit
    assert "Environment=PATH=%h/leaseholder/venvs/engine/bin:/usr/local/cuda-13.0/bin:" in unit
    # The STT unit then needs no CUDA_HOME of its own, as on the box; a 12.x toolkit is not used.
    assert "CUDA_HOME" not in (out / "systemd" / "lh-stt.service").read_text()
    out, _ = render(
        tmp_path / "old",
        "a100-40gb",
        host(cuda_toolkit={"version": "12.4", "path": "/usr/local/cuda"}),
    )
    assert f"CUDA_HOME={profile.VENV_CUDA}" in (out / "systemd" / "lh-engine.service").read_text()


def test_the_cpu_tier_shrinks_to_what_the_host_holds_and_is_refused_below_the_pool(
    tmp_path: Path,
) -> None:
    pool = 5 * GIB
    small = host(shm={"path": "/dev/shm", "total_bytes": 14 * GIB, "free_bytes": 14 * GIB})
    tier, why = profile.cpu_tier(pool, 4, small, 25)
    assert tier == 12 * GIB and "2.4 times the GPU pool" in why and "/dev/shm" in why
    tight = host(memory={"total_bytes": 64 * GIB, "available_bytes": 40 * GIB, "nodes": {}})
    assert profile.cpu_tier(pool, 4, tight, 25)[0] == 15 * GIB
    # Docker's default 64 MB /dev/shm holds no tier at all.
    docker = host(shm={"path": "/dev/shm", "total_bytes": 64 << 20, "free_bytes": 64 << 20})
    with pytest.raises(profile.ProfileError, match=r"refused.*--shm-size"):
        profile.cpu_tier(pool, 4, docker, 25)
    with pytest.raises(profile.ProfileError, match="refused"):
        render(tmp_path, "a100-40gb", docker)


def test_the_core_plan_shares_the_cpus_and_warns_below_the_floor() -> None:
    def sizes(total: int) -> dict[str, int]:
        plan, _ = profile.core_plan({"0": list(range(total))}, 0)
        return {k: len(profile.parse_cpulist(plan[k])) for k, _ in profile.CORES}

    # 34 CPUs give each component its want; more leave the rest spare; 17 give each half of it.
    assert sizes(34) == dict(profile.CORES)
    plan, warnings = profile.core_plan({"0": list(range(64))}, 0)
    assert plan["spare"] == "34-63" and warnings == []
    assert sizes(17) == {k: w // 2 for k, w in profile.CORES}
    for total in (8, 12, 16, 17, 20, 30, 33):
        got = sizes(total)
        assert sum(got.values()) == total and min(got.values()) >= 1, (total, got)
        assert all(got[k] <= w for k, w in profile.CORES), (total, got)
    # Phastform's 16 CPUs on two nodes, the GPU on node 1: below the floor, with a warning; the engine takes the
    # GPU's node and the audio servers the other, as on the box.
    plan, warnings = profile.core_plan({"0": list(range(8)), "1": list(range(8, 16))}, 1)
    assert len(warnings) == 1 and "below the single-host floor of 17" in warnings[0]
    assert plan["engine_numa"] == "1" and set(profile.parse_cpulist(plan["engine"])) <= set(
        range(8, 16)
    )
    assert plan["audio_numa"] == "0" and set(profile.parse_cpulist(plan["audio"])) <= set(range(8))
    # Fewer CPUs than components: they share.
    plan, warnings = profile.core_plan({"0": [0, 1, 2, 3]}, None)
    assert warnings and all(plan[k] for k, _ in profile.CORES)


def test_a_phastform_sized_host_renders_with_its_warning(tmp_path: Path) -> None:
    facts = host(cpus={"0": list(range(8)), "1": list(range(8, 16))})
    facts["gpu"] = facts["gpu"] | {"numa_node": 1}
    out, node = render(tmp_path, "a100-40gb", facts)
    assert node["warnings"] and "mechanics" in node["warnings"][0]
    unit = (out / "systemd" / "lh-engine.service").read_text()
    assert "NUMAMask=1\n" in unit and "# NUMA node 1 is local to the GPU (detected)." in unit
    assert "NUMAMask=0\n" in (out / "systemd" / "lh-tts.service").read_text()


def test_cpulists_round_trip() -> None:
    assert profile.parse_cpulist("0-7,16-23,30") == [*range(8), *range(16, 24), 30]
    assert profile.cpulist([3, 0, 1, 2, 8, 10, 9, 12]) == "0-3,8-10,12"
    assert profile.cpulist(profile.parse_cpulist("0-3,5")) == "0-3,5"


def test_placeholders_fill_and_an_empty_one_takes_its_line() -> None:
    values = {"a.b": "x", "a.empty": "", "a.nested": "{{a.b}}y"}
    assert profile.fill("1 {{a.b}}\n{{a.empty}}\n{{a.nested}}\n", values) == "1 x\nxy\n"
    with pytest.raises(profile.ProfileError, match=r"a\.missing"):
        profile.fill("{{a.missing}}\n", values)
    with pytest.raises(profile.ProfileError, match="--host"):
        profile.resolve(profile.load("a100-40gb"), None)


FAKES = {
    "nvidia-smi": 'echo "NVIDIA A100-SXM4-40GB, 40960, 8.0, 580.65.06, 4, 16, Disabled, 400.00, 00000000:3B:00.0, Enabled"',
    "systemctl": "echo running",
    "loginctl": "echo yes",
    "unshare": "exit 1",
    "ip": "exit 0",
    "nvcc": 'echo "Cuda compilation tools, release 13.0, V13.0.88"',
    "sudo": "exit 1",
}


def test_detect_reads_the_host_through_its_tools_and_files(tmp_path: Path, monkeypatch) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in FAKES.items():
        (bin_dir / name).write_text(f"#!/bin/sh\n{body}\n")
        (bin_dir / name).chmod(0o755)
    sysfs, procfs, rootfs, shm = (tmp_path / d for d in ("sys", "proc", "root", "shm"))
    for node, cpus in (("0", "0-7"), ("1", "8-15")):
        d = sysfs / "devices" / "system" / "node" / f"node{node}"
        d.mkdir(parents=True)
        (d / "cpulist").write_text(f"{cpus}\n")
        (d / "meminfo").write_text(
            f"Node {node} MemTotal: 98304000 kB\nNode {node} MemFree: 90000000 kB\n"
        )
    pci = sysfs / "bus" / "pci" / "devices" / "0000:3b:00.0"
    pci.mkdir(parents=True)
    (pci / "numa_node").write_text("1\n")
    thp = sysfs / "kernel" / "mm" / "transparent_hugepage"
    thp.mkdir(parents=True)
    (thp / "shmem_enabled").write_text("always within_size advise [never] deny force\n")
    (procfs / "1").mkdir(parents=True)
    (procfs / "meminfo").write_text("MemTotal: 196608000 kB\nMemAvailable: 180000000 kB\n")
    (procfs / "1" / "cgroup").write_text("0::/init.scope\n")
    (rootfs / "usr" / "local").mkdir(parents=True)
    shm.mkdir()
    for key, value in {
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "LH_SYSFS": str(sysfs),
        "LH_PROCFS": str(procfs),
        "LH_ROOTFS": str(rootfs),
        "LH_SHM": str(shm),
    }.items():
        monkeypatch.setenv(key, value)
    facts = profile.detect()
    gpu = facts["gpu"]
    assert (gpu["name"], gpu["memory_mib"], gpu["compute_cap"], gpu["driver"]) == (
        "NVIDIA A100-SXM4-40GB",
        40960,
        "8.0",
        "580.65.06",
    )
    assert (
        gpu["pcie_gen"],
        gpu["pcie_width"],
        gpu["mig"],
        gpu["power_limit_w"],
        gpu["numa_node"],
    ) == (
        4,
        16,
        "Disabled",
        400.0,
        1,
    )
    assert facts["cpus"] == {"0": list(range(8)), "1": list(range(8, 16))}
    assert facts["memory"]["available_bytes"] == 180000000 * 1024
    assert facts["memory"]["nodes"]["1"]["total_bytes"] == 98304000 * 1024
    assert facts["thp_shmem"] == "never" and facts["shm"]["path"] == str(shm)
    assert (
        facts["systemd_user"],
        facts["linger"],
        facts["userns"],
        facts["sudo"],
        facts["container"],
    ) == (
        True,
        True,
        False,
        False,
        False,
    )
    assert facts["cuda_toolkit"] == {"version": "13.0", "path": str(tmp_path.resolve())}
    # The facts render the A100's node: the GPU on node 1 takes the engine (with a /dev/shm that holds the tier,
    # where the scratch directory's file system may not).
    shm_facts = {"path": "/dev/shm", "total_bytes": 100 * GIB, "free_bytes": 100 * GIB}
    _, node = render(tmp_path, "a100-40gb", facts | {"shm": shm_facts})
    assert node["plan"]["engine_numa"] == "1"
    assert node["values"]["units.cuda_home"] == str(tmp_path.resolve())
