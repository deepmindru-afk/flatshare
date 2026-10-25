"""tests/perf_schedule.py (make perf-make-room): the cells' arithmetic, the tolerance and the check, the checked-in
baseline, and, on vLLM's real scheduler, that each steady state is steady and runs the path it is named for. Nothing
here times anything: tests/test_perf_control.py, the check's negative control, does, under make perf-make-room."""

import json
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import perf_schedule as P
import pytest

BASELINE = Path(__file__).resolve().parents[3] / "benchmarks/perf/make-room-schedule.json"


def process(state: str, off: float, on: float, off2: float | None = None, run: str = "a") -> dict:
    """A process's timed steps: 40 of each configuration's two schedulers, the first's slowest 35 of them 50% above
    the given cost, as if something else had slowed them, and the second's all 20% above it, as if it ran slower."""
    times = {"off": off, "on": on} | ({} if off2 is None else {P.NOISE: off2})
    steps = {c: [[t] * 5 + [1.5 * t] * 35, [1.2 * t] * 40] for c, t in times.items()}
    return {"run": run, "rep": 1, "state": state, "order": list(steps), "times_us": steps}


def test_a_configurations_cost_is_its_cheaper_schedulers_10th_percentile_step():
    assert P.cost([15.0] * 35 + [10.0] * 5) == 10.0  # the 4th and 5th fastest of 40
    assert P.cost(list(range(1, 41))) == 4.1  # between the 4th and 5th fastest of 40
    assert P.cheapest([[12.0] * 40, [15.0] * 35 + [10.0] * 5]) == 10.0
    assert P.cheapest([[10.5] * 40, [15.0] * 35 + [10.0] * 5]) == 10.0
    assert P.cheapest([[9.0] * 40, [15.0] * 35 + [10.0] * 5]) == 9.0


def test_a_cells_ratio_is_the_median_of_its_processes_on_off():
    procs = [
        process("B64", 100, 160, 101),
        process("B64", 200, 300, 196),
        process("B64", 110, 187, 110),
    ]
    cell = P.cells(procs)["B64"]
    assert cell["ratios"] == [0.6, 0.5, 0.7] and cell["ratio"] == 0.6  # paired in each process
    assert cell["off_us"] == 110  # the median over processes
    assert cell["change_us"] == 66 and cell["on_us"] == 176  # at that cost: on = off x (1 + ratio)
    assert cell["noises"] == [0.01, -0.02, 0.0] and cell["noise"] == 0.0
    assert "noise" not in P.cells([process("C16", 100, 150)])["C16"]


def test_the_baseline_compares_its_run_b_with_its_run_a():
    procs = [process("C16", 100, 150, 101, run) for run in "ab"]
    procs += [process("C16", 100, 160, 100, "a"), process("C16", 100, 158, 99, "b")]
    procs += [process("C16", 100, 170, 100, "a"), process("C16", 100, 162, 101, "b")]
    cell = P.baseline_cells(procs)["C16"]
    assert cell["ratios_a"] == [0.5, 0.6, 0.7] and cell["ratios_b"] == [0.5, 0.58, 0.62]
    assert cell["ratio_a"] == 0.6 and cell["ratio_b"] == 0.58
    assert cell["ratio"] == 0.59 and "ratios" not in cell  # over all six processes
    assert cell["aa"] == round(1.58 / 1.6 - 1, 4)


def test_the_tolerance_is_twice_the_worst_factor_of_either_noise():
    # The worst differed by a factor of 1/0.958: twice that is 1.0896, up to a whole percent.
    assert P.tolerance([0.01, -0.042], [0.02]) == 0.09
    assert P.tolerance([0.01], [0.042, -0.001]) == 0.09  # run against run counts too
    assert P.tolerance([0.0, 0.001], [0.0]) == 0.01
    assert P.tolerance([0.01], [0.0]) == 0.03  # 1.01 ** 2 - 1 = 0.0201, rounded up


def test_the_check_fails_a_state_whose_on_off_grew_past_the_tolerance():
    baseline = {"tolerance": 0.10, "cells": {"B64": {"ratio": 0.5}, "C16": {"ratio": 0.0}}}
    assert P.check({"B64": {"ratio": 0.64}}, baseline) == []  # 1.64 / 1.5: +9.3%
    assert P.check({"B64": {"ratio": 0.2}, "C16": {"ratio": -0.05}}, baseline) == []  # cheaper
    [failure] = P.check({"B64": {"ratio": 0.66}, "C16": {"ratio": 0.05}}, baseline)
    assert failure.startswith("B64: on/off 1.66 against the baseline's 1.50, +10.7%")
    assert P.check({"A16": {"ratio": 0.0}}, baseline) == ["A16: not in the baseline"]


@pytest.mark.parametrize("again_ratio, fails", [(0.6, False), (0.7, True)])
def test_a_failing_state_is_measured_again_and_fails_only_if_it_still_exceeds(again_ratio, fails):
    baseline = {"tolerance": 0.04, "cells": {"B64": {"ratio": 0.6}, "C16": {"ratio": 0.8}}}
    # B64: two of three processes strayed, +6.25% over the baseline's on/off; C16 within it.
    procs = [process("B64", 100, 170), process("B64", 100, 170), process("B64", 100, 160)]
    procs += [process("C16", 100, 180)] * 3
    calls = []

    def again(states):
        calls.append([s.key for s in states])
        return {"processes": [process("B64", 100, 100 * (1 + again_ratio))] * 6, "load": [{}]}

    result = {"processes": procs, "load": []}
    measured, failures, suspects = P.confirm(result, baseline, again)
    assert calls == [["B64"]] and suspects == ["B64"]  # C16 is not measured again
    assert len(measured["B64"]["ratios"]) == 9 and len(result["load"]) == 1
    assert bool(failures) == fails
    assert [f.split(":")[0] for f in failures] == (["B64"] if fails else [])


def test_a_passing_run_is_not_measured_again():
    baseline = {"tolerance": 0.04, "cells": {"B64": {"ratio": 0.6}}}
    result = {"processes": [process("B64", 100, 161)] * 3, "load": []}
    measured, failures, suspects = P.confirm(result, baseline, lambda states: pytest.fail("again"))
    assert failures == [] and suspects == [] and measured["B64"]["ratio"] == 0.61


def test_the_table_shows_off_on_the_change_and_the_ratio_against_the_baseline():
    measured = P.cells([process("B64", 1000, 1620), process("C16", 250, 450)])
    baseline = {"tolerance": 0.09, "cells": {"B64": {"ratio": 0.62}, "C16": {"ratio": 0.6}}}
    lines = P.table(measured, baseline).splitlines()
    assert lines[0].startswith("| State | Running | off µs | on µs | change µs | ratio |")
    assert lines[2] == "| B | 64 | 1,000 | 1,620 | +620 | +0.62 | +0.62 | +0.0% | ok |"
    assert lines[3].endswith("| 16 | 250 | 450 | +200 | +0.80 | +0.60 | +12.5% | costlier |")
    assert "baseline" not in P.table(measured, None)


def test_the_schedulers_take_turns_of_five_steps_in_a_rotating_order(monkeypatch):
    """measure() builds INSTANCES schedulers of each configuration and times each one's WARMUP + STEPS steps in
    turns, the order rotated each turn, keeping the steps after WARMUP; stand-in schedulers record the order."""
    seen = []
    monkeypatch.setattr(P, "load", lambda path, name: None)
    monkeypatch.setattr(
        P, "steady_state", lambda module, model, state, switches: {"switches": switches}
    )
    monkeypatch.setattr(P, "timed_step", lambda sched, state, k: seen.append((id(sched), k)) or k)
    times = P.measure(Path("m.py"), "model", P.BY_KEY["C16"], ["off", "on"])
    timed = list(range(P.WARMUP, P.WARMUP + P.STEPS))
    assert times == {"off": [timed, timed], "on": [timed, timed]}
    order = [s for i, (s, _) in enumerate(seen) if i % P.TURN == 0]  # whose turn, turn by turn
    off0, on0, off1, on1 = order[:4]
    assert len({off0, on0, off1, on1}) == 4  # a scheduler each
    assert order[:8] == [off0, on0, off1, on1, on0, off1, on1, off0]
    assert [k for s, k in seen if s == on1][:10] == list(range(10))


@pytest.mark.parametrize(
    "scheduled, preempted, steady",
    [
        ({"run0": 1, "run1": 1}, set(), True),
        ({"run0": 1}, set(), False),  # a running request left
        ({"run0": 1, "head": 64}, set(), False),  # the head was admitted
        ({"run0": 1, "w0": 64}, {"run1"}, False),  # a preemption, the count unchanged
    ],
)
def test_a_step_that_admits_or_preempts_leaves_the_steady_state(
    monkeypatch, scheduled, preempted, steady
):
    out = SimpleNamespace(num_scheduled_tokens=scheduled, preempted_req_ids=preempted)
    sched = SimpleNamespace(schedule=lambda: out, update_from_output=lambda out, model_out: None)
    monkeypatch.setattr(P, "suite", lambda: SimpleNamespace(output=lambda sched, out: None))
    state = P.State("X", "X", "B", running=2)
    if steady:
        assert P.timed_step(sched, state, 0) >= 0
    else:
        with pytest.raises(P.StateChanged, match="the state changed at step 0"):
            P.timed_step(sched, state, 0)


def test_the_states_are_the_prs_tables_rows():
    assert [s.key for s in P.STATES] == [
        "A16", "A64", "A256", "A64-hit999", "A64-nohit",
        "B16", "B64", "B256", "B256-W10", "B64-hit999",
        "C16", "C64", "C256",
    ]  # fmt: skip
    assert P.CONFIGS["on"] == {
        "max_preemptions_per_step": 2,
        "budget_cap": True,
        "budget_cap_tokens": 48,
        "reserve_tokens": 0,
    }


def test_the_json_puts_a_key_a_line_and_a_short_list_on_one():
    record = {
        "cells": {"B64": {"ratio": 0.65, "ratios_a": [0.6, 0.7]}},
        "load": [{"busy": ["x 30%"]}],
    }
    text = P.dumps(record)
    assert json.loads(text) == record
    assert text.splitlines() == [
        "{",
        ' "cells": {',
        '  "B64": {',
        '   "ratio": 0.65,',
        '   "ratios_a": [0.6, 0.7]',
        "  }",
        " },",
        ' "load": [',
        "  {",
        '   "busy": ["x 30%"]',
        "  }",
        " ]",
        "}",
    ]


def script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, P.__file__, *args], capture_output=True, text=True)


def test_a_check_without_a_readable_baseline_of_this_method_exits_2(tmp_path):
    """Exit 2 is "could not measure", which a job tells from 1, "the change costs more"."""
    proc = script("check", str(tmp_path / "missing.json"))
    assert proc.returncode == 2, proc.stderr
    assert "cannot read the baseline" in proc.stderr and "Traceback" not in proc.stderr
    (tmp_path / "broken.json").write_text("{")
    assert script("check", str(tmp_path / "broken.json")).returncode == 2
    other = json.loads(BASELINE.read_text())
    other["method"] = other["method"] | {"steps": 30}
    (tmp_path / "other.json").write_text(json.dumps(other))
    proc = script("check", str(tmp_path / "other.json"))
    assert proc.returncode == 2 and "measured otherwise" in proc.stderr


def test_an_unknown_state_exits_2():
    proc = script("check", str(BASELINE), "--states", "A16,Z9")
    assert proc.returncode == 2 and "no state Z9" in proc.stderr


def test_a_python_without_torch_or_vllms_scheduler_is_told_why():
    assert P.importable(["json", "statistics"]) is None
    assert "no_such_module_lh" in P.importable(["json", "no_such_module_lh"])


def test_a_measuring_process_that_fails_means_it_could_not_measure(tmp_path):
    with pytest.raises(P.CannotMeasure, match="A16: its measuring process failed"):
        P.run(tmp_path / "no-such-module.py", str(tmp_path), [P.BY_KEY["A16"]], ["off"], 1)


def test_the_checked_in_baseline_is_complete_and_its_numbers_agree():
    baseline = json.loads(BASELINE.read_text())
    assert {"host", "cpu", "cores", "os", "python", "torch", "vllm"} <= set(baseline["machine"])
    assert baseline["date"] and baseline["beside"] and baseline["tolerance_rule"]
    assert baseline["method"] == P.METHOD  # what a check compares against
    reps = baseline["reps"]
    assert [(load["run"], load["rep"]) for load in baseline["load"]] == [
        (run, rep) for rep in range(1, reps + 1) for run in "ab"
    ]
    assert all(len(load["before"]) == len(load["after"]) == 3 for load in baseline["load"])
    assert list(baseline["cells"]) == [s.key for s in P.STATES]
    for key, cell in baseline["cells"].items():
        a, b = cell["ratios_a"], cell["ratios_b"]
        assert len(a) == len(b) == reps and len(cell["noises"]) == 2 * reps, key
        assert cell["ratio_a"] == round(statistics.median(a), 4), key
        assert cell["ratio_b"] == round(statistics.median(b), 4), key
        assert cell["ratio"] == round(statistics.median(a + b), 4), key
        assert cell["aa"] == round(P.excess(cell["ratio_b"], cell["ratio_a"]), 4), key
        assert cell["noise"] == round(statistics.median(cell["noises"]), 4), key
        costs = P.costs(cell["off_us"], cell["ratio"])
        assert {k: cell[k] for k in costs} == costs, key
    # The tolerance is the one its rule gives from the noise it records.
    cells = baseline["cells"].values()
    noise, aa = [c["noise"] for c in cells], [c["aa"] for c in cells]
    assert baseline["tolerance"] == P.tolerance(noise, aa)
    assert baseline["noise_max"] == max(noise, key=abs) and baseline["aa_max"] == max(aa, key=abs)


# On vLLM's real scheduler: no timing, only what the states do.


@pytest.fixture(scope="module")
def module(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("vllm.v1.core.sched.scheduler")
    return P.load(P.build_module(tmp_path_factory.mktemp("make-room")), "lh_perf_make_room")


@pytest.fixture(scope="module")
def long_model(tmp_path_factory):
    from conftest import write_model

    path = tmp_path_factory.mktemp("model")
    write_model(path, max_position_embeddings=P.MAX_MODEL_LEN)
    return str(path)


def steps(module, model, key: str, config: str) -> list[int]:
    """WARMUP + STEPS steps of the state, as the measurement runs them (StateChanged if it is not steady), and in
    each the number of times make-room made room, or tried to."""
    state = P.BY_KEY[key]
    sched = P.steady_state(module, model, state, P.CONFIGS[config])
    calls = []
    make_room = sched._make_room

    def counted(*args):
        calls[-1] += 1
        return make_room(*args)

    sched._make_room = counted
    for k in range(P.WARMUP + P.STEPS):
        calls.append(0)
        P.timed_step(sched, state, k)
    return calls


@pytest.mark.parametrize("key", ["A16", "A64", "A64-nohit", "B16", "B64", "C16", "C64"])
def test_each_state_is_steady_and_runs_the_path_it_is_named_for(module, long_model, key):
    on, off = steps(module, long_model, key, "on"), steps(module, long_model, key, "off")
    assert off == [0] * len(off)  # with the switches off, make-room never runs
    if key.startswith("A"):
        assert on == [0] * len(on)  # a priority-2 head: nothing to make room for
    else:
        # B: every step evaluates make-room for the priority-0 head and preempts nobody (the state would change);
        # C: every step finds that the head fits, with no slot for it.
        assert on == [1] * len(on)
