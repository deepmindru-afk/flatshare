"""The A100 run's campaign files (benchmarks/a100): the core's five rounds and the optional arms' campaigns name arm
files that load, with the seeds and the order DESIGN.md §5.3 gives; probe_results.py reads what probe.py and tail.py
print and picks C''s stock arm by P-1's rule; and the run scripts parse."""

from __future__ import annotations

import importlib.util
import itertools
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from leaseholder_bench.e2.arm import load_arm

REPO = Path(__file__).resolve().parents[3]
A100 = REPO / "benchmarks" / "a100"
DEPLOY = REPO / "deploy"


def _module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


results = _module("lh_a100_probe_results", A100 / "probe_results.py")
PR = "make-room --additional-config.make_room.reserve_tokens 0 --additional-config.make_room.budget_cap_tokens 48"


def campaign(path: Path) -> tuple[int | None, list[Path]]:
    """A campaign file's agents seed and its arm files, as campaign.sh reads them."""
    seed, arms = None, []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if m := re.fullmatch(r"agents-seed\s+(\d+)", line):
            seed = int(m.group(1))
        else:
            arms.append(path.parent / line)
    return seed, arms


def test_the_cores_rounds_rotate_the_engines_with_no_switch_between_rounds() -> None:
    previous = PR  # block 3 leaves the PRs' values running
    w5 = {"w5-disc-pr-ab48-s24.toml", "w5-disc-frozen-s24.toml", "w5-a1-voice.toml"}
    for n in range(1, 6):
        seed, arms = campaign(A100 / f"a100-r{n}.campaign")
        assert seed == 10 + n
        assert {arm.name for arm in arms} == w5 and all(arm.parent.name == "arms" for arm in arms)
        engines = [str(load_arm(arm).engine) for arm in arms]
        assert engines[0] == previous, n
        assert engines.count(PR) == 1 and engines.count("frozen") == 2
        # The engine switches once within a round: voice alone sits beside the frozen point.
        assert sum(a != b for a, b in itertools.pairwise(engines)) == 1
        previous = engines[-1]


@pytest.mark.parametrize("path", sorted(A100.glob("*.campaign")), ids=lambda p: p.name)
def test_every_campaign_names_arm_files_that_load(path: Path) -> None:
    seed, arms = campaign(path)
    assert arms and seed is not None
    for arm in arms:
        assert arm.is_file(), arm
        load_arm(arm)


def test_a_prime_is_the_cores_arms_at_32_sessions() -> None:
    for ours, w5 in (
        ("a100-aprime-frozen-s32.toml", "w5-disc-frozen-s24.toml"),
        ("a100-aprime-pr-ab48-s32.toml", "w5-disc-pr-ab48-s24.toml"),
    ):
        a, b = load_arm(A100 / "arms" / ours), load_arm(REPO / "benchmarks" / "e2" / "arms" / w5)
        assert a.agents.sessions == 32 and b.agents.sessions == 24
        assert (a.engine, a.voice, a.agents.data, a.arm) == (
            b.engine,
            b.voice,
            b.agents.data,
            b.arm,
        )
    seeds = [campaign(A100 / f"a100-aprime-r{n}.campaign")[0] for n in (1, 2)]
    resubmit = [campaign(A100 / f"a100-resubmit-r{n}.campaign")[0] for n in (3, 4, 5)]
    assert seeds == [11, 12] and resubmit == [13, 14, 15]


PROBE_LINE = (
    "kv: priority-0 TTFT {ttft} ms, 16 tokens {t16} ms, 40 tokens {t40}, for 761 prompt tokens, sent with 2 "
    "priority-2 requests of 17999 tokens running and 24 of 2274 blocks free; after it, 2 running and 0 waiting"
)
TAIL = """tail: 12 agents with fresh prompts of 6,000-30,000 tokens beside 2 calls with 40-token replies, for 240 s
replies: 40 turns, 16 tokens p50 {r16} p90 400 p95 420 max 500 ms, 40 tokens p50 {r40} p90 917 p95 2372 max ≥30000 ms
agents: 100 requests in 240 s, 0.42 a second, with 7,000 prompt and 20 output tokens a second
calls: 40 turns, {failed} failed (counted at 30,000 ms), TTFT p50 110 p90 300 p95 {p95} max 900 ms; agents: 0 of 12 stopped early
"""


def write_runs(root: Path, p95s: dict[str, list[str]]) -> None:
    for prefix, values in p95s.items():
        block = root / ("block1" if prefix == "frozen" else "block2")
        block.mkdir(parents=True, exist_ok=True)
        for i, p95 in enumerate(values, 1):
            text = TAIL.format(r16=300, r40=900, failed=int(p95.startswith("≥")), p95=p95)
            (block / f"{prefix}-tail-{i}.txt").write_text(text)


def test_the_probes_outputs_read_back() -> None:
    probe = results.parse_probe(
        "probe: ...\n" + PROBE_LINE.format(ttft=1650, t16="1800", t40="2400 ms")
    )
    assert (probe.case, probe.ttft_ms, probe.marks) == ("kv", 1650, {16: 1800, 40: 2400})
    assert (probe.running, probe.prompt_tokens, probe.free, probe.blocks) == (2, 17999, 24, 2274)
    ended = results.parse_probe(PROBE_LINE.format(ttft=120, t16="140", t40="—"))
    assert ended.marks == {16: 140, 40: None}
    tail = results.parse_tail(TAIL.format(r16=310, r40=910, failed=1, p95="≥30000"))
    assert (tail.turns, tail.failed, tail.stopped, tail.agents) == (40, 1, 0, 12)
    assert tail.ttft["p95"] == (30000, True) and tail.ttft["p50"] == (110, False)
    assert tail.replies[40]["max"] == (30000, True) and tail.replies[16]["p50"] == (310, False)
    assert tail.requests_per_s == pytest.approx(0.42)
    with pytest.raises(ValueError):
        results.parse_tail("tail: nothing measured\n")


def test_c_primes_stock_arm_is_the_lowest_median_first_token_p95(tmp_path: Path, capsys) -> None:
    write_runs(
        tmp_path,
        {
            "frozen": ["2400", "2600"],
            "stock2048": ["1900", "≥30000"],
            "stock1536": ["2000", "2100"],
        },
    )
    name, flags, table = results.pick_stock(tmp_path, 1536)
    # 2,048/0's median is (1,900 + 30,000)/2: a failed turn counts at the timeout.
    assert name == "1536" and flags == (
        "--max-num-batched-tokens",
        "1536",
        "--long-prefill-token-threshold",
        "768",
    )
    assert [row[2] for row in table] == [2500, 15950, 2050]
    assert results.main(["pick-stock", str(tmp_path), "--cap", "1536"]) == 0
    assert (
        capsys.readouterr().out
        == "1536 --max-num-batched-tokens 1536 --long-prefill-token-threshold 768\n"
    )
    write_runs(tmp_path, {"frozen": ["1000", "1100"]})
    assert results.pick_stock(tmp_path, 1536)[:2] == ("frozen", ())
    # A setting without its tail runs is an error, not a silent pick of the others.
    assert results.main(["pick-stock", str(tmp_path), "--cap", "1280"]) == 2
    table = results.markdown(tmp_path)
    assert "| block1/frozen-tail-1 | tail (p50) | 110 | 300 | 900 | 1000 |" in table


SCRIPTS = sorted(
    [
        *A100.glob("*.sh"),
        *(DEPLOY / "a100").glob("*.sh"),
        DEPLOY / "a100" / "requirements" / "compile.sh",
    ]
)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_the_run_scripts_parse_and_refuse_a_wrong_argument(script: Path, tmp_path: Path) -> None:
    # Their runs are the CPU dry run's (test_a100_dry_run.py); here bash parses each, and each that takes arguments
    # stops at a wrong one before it touches the host.
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    if script.name in {"compile.sh", "pull.sh", "push.sh"}:
        return
    done = subprocess.run(
        ["bash", str(script), "--no-such-option"],
        capture_output=True,
        text=True,
        timeout=30,
        env=os.environ | {"HOME": str(tmp_path)},
    )
    assert done.returncode == 2 and "usage" in done.stderr, (done.returncode, done.stderr)
    assert not any(tmp_path.iterdir()), sorted(tmp_path.rglob("*"))  # nothing written first


def test_bench_marlin_imports_without_a_gpu_and_writes_its_rows(tmp_path: Path) -> None:
    bench = _module("lh_a100_bench_marlin", A100 / "bench_marlin.py")
    # The model's four linear shapes, (N, K): qkv, o, gate_up and down of Qwen3-4B-Instruct-2507.
    assert bench.SHAPES == (
        ("qkv", 6144, 2560),
        ("o", 2560, 4096),
        ("gate_up", 19456, 2560),
        ("down", 2560, 9728),
    )
    assert bench.parse_ms("1,16,1024") == [1, 16, 1024]
    with pytest.raises(Exception, match="positive"):
        bench.parse_ms("0,16")
    rows = [
        bench.Row(shape, m, kernel, us)
        for shape, *_ in bench.SHAPES
        for m, kernel, us in ((16, "marlin", 30.0), (16, "bf16", 20.0), (16, "deepgemm", None))
    ]
    rows[2].note = "RuntimeError: DeepGEMM runs on Hopper and Blackwell GPUs only"
    assert bench.ratios(rows) == {16: 1.5}
    bench.write_csv(rows, tmp_path / "gemm.csv")
    lines = (tmp_path / "gemm.csv").read_text().splitlines()
    assert lines[0] == "shape,n,k,m,kernel,us,tflops,note"
    # 2 x 16 x 6,144 x 2,560 FLOP in 30 µs.
    assert lines[1] == "qkv,6144,2560,16,marlin,30.00,16.8,"
    assert lines[3].startswith("qkv,6144,2560,16,deepgemm,,,RuntimeError")
