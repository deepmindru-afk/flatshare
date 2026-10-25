"""make-room's cost per step: Scheduler.schedule()'s wall time on vLLM 0.32.0's real scheduler, on CPU, with the
switches off and at the PRs' values, and the regression check of that cost against a measured baseline.

The module is make-room's, as build.py generates it at engine start (``--module`` times another generated module), its
switches off or at the PRs' values: ``max_preemptions_per_step 2``, ``budget_cap true``, ``budget_cap_tokens 48``,
``reserve_tokens 0``. It runs in the steady states the PRs' cost tables were measured in (#63350 and #63351), in which
nothing is admitted or preempted for the measured steps, so that every step does the same work:

  A  R running priority-2 decodes; the waiting head is a priority-2 request that does not fit the free KV (the
     saturated background tier), then W-1 small priority-2 requests behind it.
  B  as A, but the head is a priority-0 request too large for two victims: make-room evaluates every step and
     preempts nobody.
  C  every running slot taken (R = max_num_seqs); the head is a small priority-0 request that fits the KV but has no
     slot: make-room's fits path every step.

The head shares its first blocks with a running request's prompt (127 of them unless a state says otherwise), so its
prefix-cache lookup walks that many cached blocks.

A process times one state. It builds two schedulers for each configuration in that state, then the schedulers take
turns of 5 steps, 5 of warm-up and 40 timed each, so that whatever else runs on the machine meanwhile slows them
alike. A scheduler's cost is its 10th-percentile step: in a steady state every step does the same work, and the
fastest are those that nothing else slowed (nor a turn's first, cold after the others'). A configuration's cost in
the process is its cheaper scheduler's: one scheduler of a configuration may run a few percent slower than another
through the whole process, as their places in memory differ, and the pair makes that need both. A state's ratio
is the median over its processes of each one's on/off - 1: what the change costs as a fraction of the scheduler's own
step, the same on a machine that is uniformly faster or slower. Its cost off in µs is the median over the processes,
and the change's cost in µs that times the ratio, so that on = off x (1 + ratio). A run times every state in each of 3
repetitions, a process each, the states in a rotated order.

What it can see: a state fails when the change's own cost a step grows by the tolerance of the step with the switches
on (the table's "fails past", in µs), which is a small share of the change's cost where the change costs much (the B
states) and a large one where it costs little (the A states). And what it cannot: with 40 timed steps the
10th-percentile step lies between the 4th and 5th fastest, so work added to fewer than about 9 steps in 10 does not
move it. Each cell records the ratio by the median step as well, as information the check does not read.

The check fails a state whose on/off exceeds the baseline's by more than the baseline's tolerance, a fraction of it
(tolerance()), measured again over twice as many processes more and still exceeding it (confirm()). The baseline (benchmarks/perf/make-room-schedule.json) is two such runs, A and B, their repetitions
interleaved and the off configuration timed twice in each process: its noise is off against off within each process,
and run A against run B, compared as the check compares a run with the baseline.

    python tests/perf_schedule.py check BASELINE [--out JSON] [--reps 3] [--states A16,B64] [--module MODULE]
    python tests/perf_schedule.py baseline BASELINE [--reps 3] [--beside TEXT]

Both need torch and vLLM 0.32.0's dependencies, with vLLM's source on PYTHONPATH (``make perf-make-room``). The exit
status is 0 when every state is within the tolerance, 1 when one costs more than it did, and 2 when it could not
measure: no torch or vLLM's scheduler, a baseline missing, unreadable or measured otherwise, or a measuring process
that failed or left its steady state.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent / "src/leaseholder_vllm/build.py"
WARMUP, STEPS = 5, 40
TURN = 5  # the steps a scheduler takes in a row
INSTANCES = 2  # schedulers of each configuration in a process; the cheaper one's cost counts
RUN_TOKENS = 2048  # each running request's prompt
MAX_MODEL_LEN = 32768
# The PRs' values (docs/PRODUCTION.md), as make_room's keys.
CONFIGS: dict[str, dict | None] = {
    "off": None,
    "on": {
        "max_preemptions_per_step": 2,
        "budget_cap": True,
        "budget_cap_tokens": 48,
        "reserve_tokens": 0,
    },
}
# The baseline times the off configuration twice in each process, the second time as "off2": the noise.
NOISE = "off2"
# What a baseline must have been measured with for a check to compare against it.
METHOD = {
    "warmup_steps": WARMUP,
    "steps": STEPS,
    "turn_steps": TURN,
    "cost": "the 10th-percentile step of the cheaper of a configuration's schedulers in a process",
    "instances": INSTANCES,
    "configs": CONFIGS,
}


@dataclass(frozen=True)
class State:
    key: str
    label: str
    scenario: str
    running: int
    waiting: int = 1000
    hit_blocks: int = 127
    free: int = 0  # free blocks at the start; 0: the default sizing


# The rows of the PRs' cost tables, in their order.
STATES = (
    State("A16", "A: only priority 2 work, the head with a 127-block hit", "A", 16),
    State("A64", "A", "A", 64),
    State("A256", "A", "A", 256),
    State("A64-hit999", "A, the head with a 999-block hit (W = 100)", "A", 64, 100, 999, 500),
    State("A64-nohit", "A, the head with no hit (W = 100)", "A", 64, 100, 0),
    State("B16", "B: a priority 0 head that two victims cannot make room for", "B", 16),
    State("B64", "B", "B", 64),
    State("B256", "B", "B", 256),
    State("B256-W10", "B (W = 10)", "B", 256, 10),
    State("B64-hit999", "B, the head with a 999-block hit (W = 100)", "B", 64, 100, 999, 500),
    State("C16", "C: every slot taken, a priority 0 head that fits", "C", 16),
    State("C64", "C", "C", 64),
    State("C256", "C", "C", 256),
)
BY_KEY = {state.key: state for state in STATES}


class StateChanged(Exception):
    """The scheduler left the steady state during the measured steps."""


class CannotMeasure(Exception):
    """The check could not measure: the environment, the baseline or a measuring process, not the change. Exit 2."""


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


# The scheduler, as the suites build it: tests/test_make_room.py's scheduler and requests, conftest.py's model.


def suite():
    """tests/test_make_room.py, which builds the suites' schedulers and requests; it needs torch and vLLM."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import test_make_room

    return test_make_room


def load(path: Path, name: str):
    """A generated scheduler module, from its file."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def steady_state(module, model: str, state: State, switches: dict | None):
    """``module``'s Scheduler with ``switches`` in ``state``: the running requests decoding, the head and the requests
    behind it waiting."""
    T = suite()
    block = T.BLOCK
    R = state.running
    head_tokens = 32000 if state.scenario in "AB" else 1024
    long_run = max(RUN_TOKENS, block * state.hit_blocks + block)
    run_blocks = (R - 1) * cdiv(RUN_TOKENS + 64, block) + cdiv(long_run + 64, block)
    growth = R * (WARMUP + STEPS + 2) // block + R + 64
    free = state.free or max(900 if state.scenario in "AB" else 400, growth)
    sched = T.scheduler(
        module.Scheduler,
        model,
        run_blocks + free + 1,  # + vLLM's null block
        switches,
        max_num_batched_tokens=8192,
        max_model_len=MAX_MODEL_LEN,
        prefix_caching=True,
        max_num_seqs=R if state.scenario == "C" else 1024,
    )
    # R running priority-2 requests with unique prompts; the first one's is long enough to hold the head's shared
    # prefix.
    tok = 1000
    running = []
    for i in range(R):
        n = long_run if i == 0 else RUN_TOKENS
        prompt = list(range(tok, tok + n))
        running.append(T.request(f"run{i}", 0, 2, float(i), max_tokens=100_000, prompt=prompt))
        tok += n
    for req in running:
        sched.add_request(req)
    for _ in range(4000):
        if all(
            req.status == T.RequestStatus.RUNNING and not req.is_prefill_chunk for req in running
        ):
            break
        T.step(sched)
    else:
        raise StateChanged("the running requests never all decoded")
    shared = running[0].prompt_token_ids[: block * state.hit_blocks]
    suffix = list(range(tok, tok + head_tokens - len(shared)))
    tok += len(suffix)
    priority = 2 if state.scenario == "A" else 0
    sched.add_request(T.request("head", 0, priority, 1e6, max_tokens=64, prompt=shared + suffix))
    for j in range(state.waiting - 1):
        prompt = list(range(tok, tok + 64))
        sched.add_request(T.request(f"w{j}", 0, 2, 1e6 + 1 + j, max_tokens=64, prompt=prompt))
        tok += 64
    return sched


def timed_step(sched, state: State, k: int) -> float:
    """schedule()'s wall time in µs for one step, its output fed back as a synchronous engine does; StateChanged if
    the step schedules other than the running requests or preempts."""
    t0 = time.perf_counter_ns()
    out = sched.schedule()
    t1 = time.perf_counter_ns()
    sched.update_from_output(out, suite().output(sched, out))
    if (
        len(out.num_scheduled_tokens) != state.running
        or "head" in out.num_scheduled_tokens
        or out.preempted_req_ids
    ):
        raise StateChanged(
            f"{state.key}: the state changed at step {k}: "
            f"{len(out.num_scheduled_tokens)} scheduled, {len(out.preempted_req_ids or ())} preempted"
        )
    return (t1 - t0) / 1e3


def measure(
    module_path: Path, model: str, state: State, configs: list[str]
) -> dict[str, list[list[float]]]:
    """One process's measurement: INSTANCES schedulers of their own for each configuration in ``configs``, all in
    ``state``, then WARMUP and STEPS steps of each, the schedulers taking turns of TURN steps in an order rotated each
    turn. Returns each configuration's timed steps in µs, scheduler by scheduler."""
    module = load(module_path, module_path.stem)
    order = [(c, i) for i in range(INSTANCES) for c in configs]
    scheds = {
        key: steady_state(module, model, state, CONFIGS["off" if key[0] == NOISE else key[0]])
        for key in order
    }
    gc.collect()
    gc.freeze()  # the states' objects: the collector need not walk them while steps are timed
    times: dict[tuple[str, int], list[float]] = {key: [] for key in order}
    for turn, first in enumerate(range(0, WARMUP + STEPS, TURN)):
        for key in rotated(order, turn):
            for k in range(first, min(first + TURN, WARMUP + STEPS)):
                t = timed_step(scheds[key], state, k)
                if k >= WARMUP:
                    times[key].append(t)
    gc.unfreeze()
    return {c: [times[c, i] for i in range(INSTANCES)] for c in configs}


# The runs: every state in each repetition, a process each, the states and configurations rotated.


def rotated(items: list, k: int) -> list:
    k %= len(items)
    return items[k:] + items[:k]


def busy() -> list[str]:
    """What else uses the CPU: the commands whose processes use 25% of a CPU or more between them, as ps reports it
    (on Linux, each process's use averaged over its life), this process left out."""
    proc = subprocess.run(["ps", "-eo", "pid=,pcpu=,comm="], capture_output=True, text=True)
    use: dict[str, float] = {}
    for line in proc.stdout.splitlines():
        fields = line.split(None, 2)
        if len(fields) == 3 and int(fields[0]) != os.getpid():
            name = Path(fields[2].strip()).name
            use[name] = use.get(name, 0.0) + float(fields[1])
    top = sorted(use.items(), key=lambda item: -item[1])
    return [f"{name} {cpu:.0f}%" for name, cpu in top if cpu >= 25]


def run(
    module_path: Path,
    model: str,
    states: list[State],
    configs: list[str],
    reps: int,
    runs: str = "a",
) -> dict:
    """``reps`` repetitions of each run in ``runs`` (their repetitions interleaved), every state in ``states`` in
    each, a process each, timing ``configs``. Returns each process's timed steps by configuration, and for each
    repetition the load average before and after it and what else used the CPU as it started."""
    processes, load = [], []
    # The scheduler is Python; torch's thread pools would only take CPUs from what else runs.
    env = os.environ | {
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    for rep in range(reps):
        for r, name in enumerate(runs):
            k = rep * len(runs) + r
            before, others = os.getloadavg(), busy()
            for j, state in rotated(list(enumerate(states)), k):
                order = rotated(configs, k + j)
                cmd = [sys.executable, str(Path(__file__).resolve()), "measure", str(module_path)]
                cmd += [model, state.key, *order]
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
                if proc.returncode:
                    raise CannotMeasure(
                        f"{state.key}: its measuring process failed:\n{proc.stderr[-2000:]}"
                    )
                times = json.loads(proc.stdout)
                processes.append(
                    {
                        "run": name,
                        "rep": rep + 1,
                        "state": state.key,
                        "order": order,
                        "times_us": times,
                    }
                )
                where = f"{rep + 1}{name if len(runs) > 1 else ''}/{reps}"
                costs = "  ".join(f"{c} {cheapest(times[c]):,.0f}" for c in configs)
                print(f"  {where} {state.key:<11} {costs}", file=sys.stderr, flush=True)
            after = os.getloadavg()
            load.append(
                {
                    "run": name,
                    "rep": rep + 1,
                    "before": [round(x, 2) for x in before],
                    "after": [round(x, 2) for x in after],
                    "busy": others,
                }
            )
    return {"processes": processes, "load": load}


# Costs, ratios, the tolerance and the check: arithmetic on the timed steps, testable without torch.


def cost(times: list[float]) -> float:
    """A scheduler's cost: the 10th percentile of its timed steps."""
    return statistics.quantiles(times, n=10)[0]


def cheapest(schedulers: list[list[float]]) -> float:
    """A configuration's cost in a process: its cheaper scheduler's."""
    return min(map(cost, schedulers))


def cells(processes: list[dict]) -> dict[str, dict]:
    """Each state's cell from its processes, in STATES' order: the ratio, the median over the processes of each
    one's on/off - 1, which its turns pair; the cost off, the median over the processes, in µs, and the change's at
    that cost; where the off configuration was timed twice, the noise, the median of off2/off - 1; and, as
    information the check does not read, the same ratio by the median step, which work on fewer than about 9 steps
    in 10 moves and the 10th-percentile step does not."""
    by_state: dict[str, list[tuple[dict[str, float], dict[str, float]]]] = {}
    for proc in processes:
        p10 = {c: cheapest(t) for c, t in proc["times_us"].items()}
        median = {c: min(map(statistics.median, t)) for c, t in proc["times_us"].items()}
        by_state.setdefault(proc["state"], []).append((p10, median))
    out = {}
    for key in (s.key for s in STATES if s.key in by_state):
        procs = [p10 for p10, _ in by_state[key]]
        ratios = [round(p["on"] / p["off"] - 1, 4) for p in procs]
        cell: dict = {"off_us": round(statistics.median(p["off"] for p in procs), 1)}
        cell |= costs(cell["off_us"], round(statistics.median(ratios), 4))
        cell["ratios"] = ratios
        cell["median_ratio"] = round(
            statistics.median(m["on"] / m["off"] - 1 for _, m in by_state[key]), 4
        )
        if NOISE in procs[0]:
            cell["noises"] = [round(p[NOISE] / p["off"] - 1, 4) for p in procs]
            cell["noise"] = round(statistics.median(cell["noises"]), 4)
        out[key] = cell
    return out


def costs(off_us: float, ratio: float) -> dict[str, float]:
    """A state's costs from its cost off and its ratio: on = off x (1 + ratio), the change on - off."""
    change = round(off_us * ratio, 1)
    return {"on_us": round(off_us + change, 1), "change_us": change, "ratio": ratio}


def excess(ratio: float, base: float) -> float:
    """How far a state's on/off exceeds the baseline's, as a fraction of the baseline's."""
    return (1 + ratio) / (1 + base) - 1


def baseline_cells(processes: list[dict]) -> dict[str, dict]:
    """The baseline's cells: over both runs' processes, with each run's ratios and run B's excess over run A."""
    out = cells(processes)
    by_run = {name: cells([p for p in processes if p["run"] == name]) for name in "ab"}
    for key, cell in out.items():
        a, b = by_run["a"][key], by_run["b"][key]
        del cell["ratios"]
        cell |= {"ratios_a": a["ratios"], "ratios_b": b["ratios"]}
        cell |= {"ratio_a": a["ratio"], "ratio_b": b["ratio"]}
        cell["aa"] = round(excess(b["ratio"], a["ratio"]), 4)
    return out


def tolerance(noise: list[float], aa: list[float]) -> float:
    """The check's tolerance: twice the largest factor by which the baseline's two measurements of something
    unchanged differed in any state, as a fraction rounded up to a whole percent. Those are off against off in each
    process (``noise``: the median over the processes of off2/off - 1) and its run B against its run A (``aa``: the
    excess, as the check compares on/off), and a check run and the baseline may each be off by that much."""
    worst = max(abs(math.log1p(x)) for x in [*noise, *aa])
    return math.ceil(round(100 * math.expm1(2 * worst), 6)) / 100


def check(measured: dict[str, dict], baseline: dict) -> list[str]:
    """The states in ``measured`` whose on/off exceeds the baseline's by more than its tolerance, and what they cost;
    a state the baseline lacks fails too."""
    failures = []
    for key, cell in measured.items():
        base = baseline["cells"].get(key)
        if base is None:
            failures.append(f"{key}: not in the baseline")
            continue
        over = excess(cell["ratio"], base["ratio"])
        if over > baseline["tolerance"]:
            failures.append(
                f"{key}: on/off {1 + cell['ratio']:.2f} against the baseline's "
                f"{1 + base['ratio']:.2f}, {over:+.1%} (tolerance {baseline['tolerance']:.0%})"
            )
    return failures


def confirm(result: dict, baseline: dict, again) -> tuple[dict[str, dict], list[str], list[str]]:
    """The check of ``result``'s processes against ``baseline``, a state that fails measured again first: ``again``
    runs the failing states for twice as many processes more, and a state fails only if its ratio over all of its
    processes still exceeds the tolerance. Two of an unchanged module's three processes can stray the same way, and
    rarely do; a costlier module fails again. Returns the cells, the failures and the states measured again."""
    measured = cells(result["processes"])
    failing = {f.split(":")[0] for f in check(measured, baseline) if " on/off " in f}
    suspects = [key for key in measured if key in failing]
    if suspects:
        more = again([BY_KEY[key] for key in suspects])
        result["processes"] += more["processes"]
        result["load"] += more["load"]
        measured = cells(result["processes"])
    return measured, check(measured, baseline), suspects


def table(measured: dict[str, dict], baseline: dict | None) -> str:
    """The cells as a Markdown table, in the PRs' rows' order. Against a baseline, "fails past" is how many µs a step
    more the change may cost before the state fails: the tolerance of the step with the switches on."""
    head = "| State | Running | off µs | on µs | change µs | ratio |"
    rule = "|---|---:|---:|---:|---:|---:|"
    if baseline:
        head += " baseline | excess | fails past | |"
        rule += "---:|---:|---:|---|"
    lines = [head, rule]
    for state in STATES:
        cell = measured.get(state.key)
        if cell is None:
            continue
        line = (
            f"| {state.label} | {state.running} | {cell['off_us']:,.0f} | {cell['on_us']:,.0f} "
            f"| {cell['change_us']:+,.0f} | {cell['ratio']:+.2f} |"
        )
        if baseline:
            base = baseline["cells"].get(state.key)
            if base is None:
                line += " — | — | not in the baseline |"
            else:
                over = excess(cell["ratio"], base["ratio"])
                verdict = "costlier" if over > baseline["tolerance"] else "ok"
                # The change may cost this much more a step before the state fails.
                room = cell["off_us"] * baseline["tolerance"] * (1 + base["ratio"])
                line += f" {base['ratio']:+.2f} | {over:+.1%} | +{room:,.0f} µs | {verdict} |"
        lines.append(line)
    return "\n".join(lines)


def dumps(value, indent: int = 0) -> str:
    """JSON with one key of an object a line and a list of numbers or strings on one line."""
    pad = " " * (indent + 1)
    if isinstance(value, dict) and value:
        items = [f"{pad}{json.dumps(k)}: {dumps(v, indent + 1)}" for k, v in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    if isinstance(value, list) and any(isinstance(v, dict | list) for v in value):
        items = [pad + dumps(v, indent + 1) for v in value]
        return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"
    return json.dumps(value, ensure_ascii=False)


# The machine and the environment, for the record.


def machine() -> dict:
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        if sys.platform == "darwin":
            proc = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            )
            cpu = proc.stdout.strip() or cpu
    import torch
    import vllm

    src = Path(vllm.__file__).resolve().parents[1]
    proc = subprocess.run(
        ["git", "-C", str(src), "describe", "--tags", "--always"], capture_output=True, text=True
    )
    return {
        "host": platform.node().split(".")[0],
        "cpu": cpu,
        "cores": os.cpu_count(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "vllm": proc.stdout.strip() or str(src),
    }


def importable(modules: list[str]) -> str | None:
    """None if this Python imports ``modules``, or why not."""
    for name in modules:
        try:
            importlib.import_module(name)
        except ImportError as error:
            return f"{name}: {error}"
    return None


def read_baseline(path: Path) -> dict:
    """The baseline at ``path``, measured as this check measures; CannotMeasure if it is not one."""
    try:
        baseline = json.loads(path.read_text())
        missing = [
            k for k in ("date", "machine", "method", "tolerance", "cells") if k not in baseline
        ]
    except (OSError, ValueError, TypeError) as error:
        raise CannotMeasure(f"cannot read the baseline {path}: {error}") from error
    if missing:
        raise CannotMeasure(f"cannot read the baseline {path}: it has no {', '.join(missing)}")
    if baseline["method"] != METHOD:
        raise CannotMeasure(
            f"{path} was measured otherwise than this check measures: measure a new baseline "
            "(make perf-make-room UPDATE=1)"
        )
    return baseline


def machine_check(here: dict, baseline: dict, any_machine: bool) -> str | None:
    """Whether a check here may compare with ``baseline``: the ratios depend on the CPU, Python and torch, so another's
    raises CannotMeasure, or with ``any_machine`` returns a warning to print. None when they agree."""
    there = baseline["machine"]
    keys = [k for k in ("cpu", "python", "torch") if here.get(k) != there.get(k)]
    if not keys:
        return None
    diffs = "; ".join(f"{k}: {there.get(k)} there, {here.get(k)} here" for k in keys)
    what = f"the baseline is {there.get('host')}'s and this machine differs ({diffs})"
    if any_machine:
        return f"against another machine's baseline: {what}; ratios are CPU-specific, so read it with care"
    raise CannotMeasure(
        f"{what}: ratios are CPU-specific, so measure one here (make perf-make-room UPDATE=1 "
        "PERF_BASELINE=build/perf-baseline.json PERF_BESIDE=...) and check against it, or pass --any-machine"
    )


def build_module(out: Path) -> Path:
    """make-room's module, as build.py writes it at engine start, from the vLLM source on PYTHONPATH."""
    build = load(BUILD, "lh_perf_build")
    return build.build("make-room", out, source_tree=True)


def main() -> None:
    try:
        sys.exit(command())
    except CannotMeasure as error:
        print(f"perf_schedule.py: {error}", file=sys.stderr)
        sys.exit(2)


def command() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "baseline"):
        p = sub.add_parser(name)
        p.add_argument("baseline", type=Path)
        p.add_argument("--reps", type=int, default=3)
        p.add_argument("--states", default=",".join(BY_KEY), help="comma-separated states")
        p.add_argument("--module", type=Path, help="a generated module in place of make-room's")
        p.add_argument("--raw", type=Path, help="every process's timed steps, as JSON")
    sub.choices["check"].add_argument("--out", type=Path, help="the run's JSON")
    sub.choices["check"].add_argument(
        "--any-machine", action="store_true", help="check against another machine's baseline"
    )
    sub.choices["baseline"].add_argument("--beside", default="", help="what else ran meanwhile")
    m = sub.add_parser("measure")  # one process's measurement, run by run()
    m.add_argument("module", type=Path)
    m.add_argument("model")
    m.add_argument("state", choices=list(BY_KEY))
    m.add_argument("configs", nargs="+", choices=[*CONFIGS, NOISE])
    args = parser.parse_args()

    if args.command == "measure":
        times = measure(args.module, args.model, BY_KEY[args.state], args.configs)
        print(json.dumps({c: [[round(t, 2) for t in s] for s in v] for c, v in times.items()}))
        return 0

    unknown = [key for key in args.states.split(",") if key not in BY_KEY]
    if unknown:
        raise CannotMeasure(f"no state {', '.join(unknown)}; the states are {', '.join(BY_KEY)}")
    states = [BY_KEY[key] for key in args.states.split(",")]
    if args.command == "baseline" and not args.beside.strip():
        raise CannotMeasure(
            "a baseline needs --beside: what else ran on the machine as it was measured"
        )
    baseline = read_baseline(args.baseline) if args.command == "check" else None
    why = importable(["pytest", "torch", "vllm.v1.core.sched.scheduler"])
    if why:
        raise CannotMeasure(f"this Python cannot import what the scheduler needs ({why})")
    here = machine()
    warning = machine_check(here, baseline, args.any_machine) if baseline else None
    if warning:
        print(f"perf_schedule.py: {warning}", file=sys.stderr)
    from conftest import write_model

    configs = ["off", "on"] + ([NOISE] if args.command == "baseline" else [])
    runs = "ab" if args.command == "baseline" else "a"
    with tempfile.TemporaryDirectory() as tmp:
        module = args.module or build_module(Path(tmp))
        model = Path(tmp) / "model"
        model.mkdir()
        write_model(model, max_position_embeddings=MAX_MODEL_LEN)
        started = datetime.now(UTC)
        result = run(module, str(model), states, configs, args.reps, runs)
        if baseline is not None:

            def again(failing: list[State]) -> dict:
                keys = ", ".join(s.key for s in failing)
                print(f"  measuring {keys} again, {2 * args.reps} processes each", file=sys.stderr)
                return run(module, str(model), failing, configs, 2 * args.reps, "c")

            measured, failures, suspects = confirm(result, baseline, again)
    if args.raw:
        args.raw.write_text(json.dumps(result) + "\n")
    record = {
        "machine": here,
        "date": started.strftime("%Y-%m-%dT%H:%MZ"),
        "module": "make-room" if args.module is None else args.module.name,
        "method": METHOD,
        "reps": args.reps,
        "load": result["load"],
    }

    if args.command == "baseline":
        measured = baseline_cells(result["processes"])
        noise = [cell["noise"] for cell in measured.values()]
        aa = [cell["aa"] for cell in measured.values()]
        record = {
            "what": "make-room's cost per step, Scheduler.schedule()'s wall time on vLLM 0.32.0's real scheduler "
            "on CPU with the switches off and at the PRs' values, written by "
            "python/leaseholder_vllm/tests/perf_schedule.py baseline",
            **record,
            "beside": args.beside,
            "tolerance": tolerance(noise, aa),
            "tolerance_rule": " ".join(tolerance.__doc__.split()),
            "noise_max": max(noise, key=abs),
            "aa_max": max(aa, key=abs),
            "cells": measured,
        }
        print(table(measured, None))
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(dumps(record) + "\n")
        print(
            f"\nbaseline written to {args.baseline}: the worst off-against-off noise "
            f"{record['noise_max']:+.1%}, the worst run B against run A {record['aa_max']:+.1%}, "
            f"so a tolerance of {record['tolerance']:.0%}"
        )
        return 0

    record |= {
        "cells": measured,
        "measured_again": suspects,
        "baseline": {
            "date": baseline["date"],
            "host": baseline["machine"]["host"],
            "tolerance": baseline["tolerance"],
        },
        "failures": failures,
    } | ({"machine_differs": warning} if warning else {})
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(dumps(record) + "\n")
    print(table(measured, baseline))
    print(
        f"\nagainst the baseline of {baseline['machine']['host']}, {baseline['date']} "
        f"(tolerance {baseline['tolerance']:.0%}):"
    )
    if failures:
        print(f"FAIL: the change costs more than it did in {len(failures)} states:")
        print("\n".join(f"  {failure}" for failure in failures))
        return 1
    print(f"ok: {len(measured)} states within it")
    return 0


if __name__ == "__main__":
    main()
