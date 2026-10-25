"""The perf check's negative control, which make perf-make-room runs after the check (LH_PERF=1): make-room built with
two of #63350's optimizations undone fails the check against the checked-in baseline in the states where that work
runs, and make-room itself passes it there, each timed in one process per state (a state that fails is measured
again over two more).

Undone, each step that tries to make room sums the decodes' growth: in state B, where two victims cannot make room
(44937b87a gave up before the sum), and in state C, where the head fits (e87503f22 checked the fit against a bound
first). The control times the scheduler, as the check does, so CI never runs it."""

import json
import os
import subprocess
import sys
from pathlib import Path

import perf_schedule as P
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("LH_PERF"), reason="it times the scheduler: make perf-make-room runs it"
)

BASELINE = Path(__file__).resolve().parents[3] / "benchmarks/perf/make-room-schedule.json"
SLOWER = (
    (  # give up before counting growth: never
        "            num_blocks_short > 0\n            and not self.need_mamba_block_aligned_split\n",
        "            False\n            and not self.need_mamba_block_aligned_split\n",
    ),
    (  # check the fit against a bound first: never
        "            self._max_growth_blocks is not None\n            and num_blocks_short",
        "            False\n            and num_blocks_short",
    ),
)
STATES = ("B64", "C256")


def check(module: Path, out: Path) -> subprocess.CompletedProcess:
    cmd = [sys.executable, P.__file__, "check", str(BASELINE), "--reps", "1"]
    cmd += ["--states", ",".join(STATES), "--module", str(module), "--out", str(out)]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_a_module_made_slower_fails_the_check_and_make_room_passes_it(tmp_path):
    pytest.importorskip("torch")
    real = P.build_module(tmp_path / "make-room")
    source = real.read_text()
    for old, new in SLOWER:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    slower = tmp_path / "slower" / real.name
    slower.parent.mkdir()
    slower.write_text(source)

    proc = check(slower, tmp_path / "slower.json")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    failures = json.loads((tmp_path / "slower.json").read_text())["failures"]
    assert sorted(failure.split(":")[0] for failure in failures) == sorted(STATES), failures
    print("\nthe negative control, make-room with two of #63350's optimizations undone, fails:")
    print("\n".join(f"  {failure}" for failure in failures))

    proc = check(real, tmp_path / "make-room.json")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    print(f"and make-room passes it at {' and '.join(STATES)}")
