"""The A100 run's analysis (docs/HARDWARE.md, the analysis), on a CPU, from a single host's records as
deploy/a100/pull.sh copies them.

    python3 benchmarks/a100/analysis.py report HOST_DIR --corpus WEKA_X32 --lh-sim LH_SIM --out OUT [--lh-bench BIN]
    python3 benchmarks/a100/analysis.py calibration-check EXPORTS --corpus WEKA_X32 --lh-sim LH_SIM --out OUT

`report` writes OUT/a100.md (`make reproduce-a100`), each section from what HOST_DIR holds and saying what it lacks,
against the forecast for the card the records name (node/host.json's GPU): benchmarks/a100/forecast-80gb.md for an
A100 80 GB, benchmarks/a100/forecast.md (the A100 40 GB's, the card the kit was built for) otherwise; a100.md says
which.
1. The step model: crates/lh-sim/scripts/fit_step_cost.py on the calibration point (runs/a100/calibration-s16) with
   the card's own E0 (runs/a100/block1/frozen-e0), each fitted term beside the forecast's central value and range
   as a ratio, and the largest miss with what it is likely to mean.
2. The calibration table: lh-sim with those costs (and E0's tier fits) against the measured point, as the box's
   (crates/lh-sim/scripts/calibration.py, whose functions it uses): requests, resume TTFT p50 / p90 / p99 and the
   GPU, CPU and computed shares of the prompt tokens.
3. The probes (probe_results.py): every run, then the KV case's wait and the prefill case's token gap on stock and on
   the PRs' values, and the calls' tail at each stock setting, beside the forecast and the box.
4. E2 within the A100: `lh-bench e2 report` over the core's points (results/e2/a100-r1 to -r5) into OUT/e2, its frontier
   quoted, and each configuration's measured running share against the forecast's.
5. Across cards: the regime coordinates the box measured and the forecast gives the A100, beside the A100's measured
   running share, labeled different hardware.

`calibration-check` is P-1's prep 6: report's calibration code on the box's three W2 operating points (the
calibration exports, calibration.py's EXPORTS) with the box's own E0 gives calibration.py report's rows for them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRIPTS = REPO / "crates" / "lh-sim" / "scripts"
FORECAST = HERE / "forecast.md"
"""The forecast for the A100 40 GB, the card the kit was built for (make forecast-a100)."""
FORECAST_80GB = HERE / "forecast-80gb.md"
"""The forecast for the A100 SXM4 80 GB (make forecast-a100 GPU=a100-80gb)."""
DASH = "\u2013"  # the range dash, as calibration.py writes it


def _module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


calibration = _module("lh_calibration", SCRIPTS / "calibration.py")
results = _module("lh_a100_probe_results", HERE / "probe_results.py")

TERMS = (
    ("step_ms", "Step: fixed", "ms", "the step's launches and kernel tails beyond the weight read"),
    (
        "us_per_token",
        "per token",
        "µs",
        "the linear layers' rate on this card's path (Marlin W8A16 on sm80)",
    ),
    ("ns_per_pair", "per pair", "ns", "FlashAttention 2's efficiency at these shapes"),
    ("ns_per_context", "per KV token read", "ns", "decode attention's bandwidth"),
)
"""fit_step_cost.py's terms, the forecast table's columns, their unit, and what a miss in each points at."""
BOX = {
    "kv_stock_ms": f"4,923{DASH}5,077 ms (Gray, the second 5060 Ti)",
    "kv_pr_ms": "225 and 297 ms (Gray)",
    "prefill_stock_gap_ms": "69 ms (Gray)",
    "prefill_pr_gap_ms": "17 ms (Gray)",
}
"""The box's probe results, measured (docs/BENCHMARKS.md; benchmarks/results/2026-10-23-make-room-probes)."""


def gpu_name(host: Path) -> str | None:
    """The card the records name: node/host.json's GPU, as nvidia-smi gives it ("NVIDIA A100-SXM4-80GB")."""
    try:
        return json.loads((host / "node" / "host.json").read_text())["gpu"]["name"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def forecast_for(gpu: str | None) -> Path:
    """The forecast for the card: forecast-80gb.md for an A100 80 GB, forecast.md (the A100 40 GB's) otherwise."""
    name = (gpu or "").upper().replace(" ", "")
    return FORECAST_80GB if "A100" in name and "80GB" in name else FORECAST


def forecast_card(text: str) -> str:
    """The card a forecast is for, as its title names it ("A100 40 GB")."""
    title = re.match(r"# Forecast: the (.+) against the frozen box\n", text)
    if title is None:
        raise ValueError("the forecast has no title naming its card")
    return title.group(1)


def forecast_costs(text: str) -> dict[str, dict[str, float]]:
    """The forecast's costs on its card for the core's model, by setting (low, central, high), in the fit's units."""
    card = forecast_card(text)
    section = text.split("## Step and tier costs, Qwen3-4B-Instruct-2507-FP8", 1)[1].split(
        "\n## ", 1
    )[0]
    out: dict[str, dict[str, float]] = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 6 and cells[0] == card and cells[1] in ("low", "central", "high"):
            values = [float(re.match(r"[\d.]+", c).group().replace(",", "")) for c in cells[2:6]]
            out[cells[1]] = dict(zip((name for name, *_ in TERMS), values, strict=True))
    if set(out) != {"low", "central", "high"}:
        raise ValueError(f"the forecast has no low, central and high cost rows for the {card}")
    return out


def forecast_probes(text: str, card: str) -> dict[str, str]:
    """The forecast's probe cases on its card, each "central ms (low-high)" as the forecast writes them."""
    on = rf"On the {re.escape(card)}: "
    kv = re.search(
        on
        + rf"(\S+) ms \(low{DASH}high (\S+)\)\. make-room's first token, [^:]*: (\S+) ms \((\S+)\)\.",
        text,
    )
    prefill = re.search(on + r"(\S+) ms \((\S+)\) and (\S+) ms \((\S+)\), with the chunks", text)
    if kv is None or prefill is None:
        raise ValueError(f"the forecast has no KV case or prefill case for the {card}")

    def pair(m: re.Match[str], i: int) -> str:
        return f"{m.group(i)} ms ({m.group(i + 1)})"

    return {
        "kv_stock": pair(kv, 1),
        "kv_pr": pair(kv, 3),
        "prefill_stock": pair(prefill, 1),
        "prefill_pr": pair(prefill, 3),
    }


def forecast_core(text: str, card: str) -> dict[str, str]:
    """The forecast's core scenario on its card, central: its running share, busy and stalled shares."""
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 11 and cells[:2] == ["A: the core, the box's pool", f"{card} (central)"]:
            return {"running": cells[8].split(" / ")[0], "busy": cells[9], "stalled": cells[10]}
    raise ValueError(f"the forecast has no central core cell for the {card}")


@dataclass(frozen=True)
class Forecast:
    """What the report sets beside the measurements, from one forecast."""

    path: Path
    card: str
    costs: dict[str, dict[str, float]]
    probes: dict[str, str]
    core: dict[str, str]


def read_forecast(path: Path) -> Forecast:
    text = path.read_text()
    card = forecast_card(text)
    return Forecast(
        path, card, forecast_costs(text), forecast_probes(text, card), forecast_core(text, card)
    )


def forecast_line(gpu: str | None, forecast: Forecast) -> str:
    """Which forecast the report compares against, and why."""
    which = f"It compares against {_shown(forecast.path)}, the forecast for the {forecast.card}"
    if gpu is None:
        return which + ": the records have no node/host.json naming their card."
    if forecast.path == FORECAST_80GB:
        return which + f", as node/host.json names the card {gpu}."
    return which + f", as node/host.json names the card {gpu}, not an A100 80 GB."


def _shown(path: Path) -> str:
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def fit(export: Path, log: Path, e0: Path | None, out: Path) -> tuple[dict[str, Any], str]:
    """fit_step_cost.py on one run: the fitted costs and the fit's report."""
    args = [sys.executable, str(SCRIPTS / "fit_step_cost.py")]
    if e0 is not None:
        args += ["--e0", str(e0)]
    done = subprocess.run(
        [*args, str(export / calibration.FILES[0]), str(log)],
        check=True,
        capture_output=True,
        text=True,
    )
    out.write_text(done.stdout)
    return json.loads(done.stdout), done.stderr


def step_model_lines(steps: dict[str, Any], forecast: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        f"| Term | Fitted | Forecast, central | Range (low{DASH}high) | Fitted over central |",
        "|---|---:|---:|---:|---:|",
    ]
    misses = []
    for name, label, unit, cause in TERMS:
        value, central = steps[name], forecast["central"][name]
        low, high = sorted((forecast["low"][name], forecast["high"][name]))
        ratio = value / central
        inside = low <= value <= high
        misses.append((abs(math.log(ratio)), label, ratio, cause, inside))
        lines.append(
            f"| {label} | {value:g} {unit} | {central:g} {unit} | {low:g}{DASH}{high:g} {unit} | {ratio:.2f}"
            + ("" if inside else ", outside the range")
            + " |"
        )
    _, label, ratio, cause, inside = max(misses)
    lines += [
        "",
        f"The largest miss is the {label.lower().removeprefix('step: ')} term, {ratio:.2f} times the forecast"
        + (" (within its range)" if inside else " (outside its range)")
        + f": {cause}.",
    ]
    return lines


def calibration_lines(
    sim: Any, export: Path, sessions: int, options: list[str], name: str
) -> tuple[list[str], list[float]]:
    """The measured point and lh-sim's contention and idle models of it, as calibration.py's table rows."""
    m = calibration.measured(export)
    lines = [
        f"| {sessions} sessions | measured | {calibration.n(m['requests'])} | "
        f"{' / '.join(map(calibration.n, m['resume']))} | {calibration.shares(m['shares'])} |"
    ]
    contention: list[float] = []
    for model, extra in (("idle (v0)", ["--service", "idle"]), ("contention", [])):
        s = sim.run(f"{name}-{model.split()[0]}", "--concurrency", str(sessions), *options, *extra)
        requests, resume, sim_shares = calibration.sim_row(s)
        lines.append(
            f"| | {model} | {calibration.n(requests)} "
            f"({calibration.signed(100 * (requests / m['requests'] - 1), 1)}%) | "
            f"{' / '.join(map(calibration.n, resume))} ({calibration.errors(resume, m['resume'])}%) | "
            f"{calibration.shares(sim_shares)} |"
        )
        if model == "contention":
            contention = resume
    return lines, contention


TABLE_HEAD = [
    "| Point | | Requests | Resume TTFT p50 / p90 / p99 (ms) | GPU / CPU / computed (%) |",
    "|---|---|---|---|---|",
]


def median_range(values: list[float]) -> str:
    if not values:
        return "not measured"
    if len(values) == 1:
        return f"{values[0]:,.0f} ms"
    return f"{statistics.median(values):,.0f} ms ({min(values):,.0f}{DASH}{max(values):,.0f}, {len(values)} runs)"


def probe_lines(runs: Path, forecast: Forecast | None = None) -> list[str]:
    """The probes' runs, and the cases beside the forecast (the A100 40 GB's unless given) and the box."""
    forecast = forecast or read_forecast(FORECAST)
    f = forecast.probes

    def probes(block: str, prefix: str) -> list[Any]:
        return [
            r for r in results.runs(runs, block, prefix).values() if isinstance(r, results.Probe)
        ]

    def gap(run: Any) -> float | None:
        t16 = run.marks.get(16)
        return None if t16 is None else (t16 - run.ttft_ms) / 15

    lines = [results.markdown(runs).rstrip("\n"), ""]
    kv_stock = [r.ttft_ms for r in probes("block1", "frozen-kv")]
    kv_pr = [r.ttft_ms for r in probes("block3", "pr-kv")]
    kv_2048 = [r.ttft_ms for r in probes("block2", "stock2048-kv")]
    pf_stock = [g for r in probes("block1", "frozen-prefill") if (g := gap(r)) is not None]
    pf_pr = [g for r in probes("block3", "pr-prefill") if (g := gap(r)) is not None]
    lines += [
        f"| Case | {forecast.card}, measured | Forecast, central (low{DASH}high) [modeled] | The box [measured] |",
        "|---|---|---|---|",
        f"| KV case, stock: the voice turn's wait | {median_range(kv_stock)} | {f['kv_stock']} "
        f"| {BOX['kv_stock_ms']} |",
        f"| KV case, stock at 2,048/0 | {median_range(kv_2048)} | | |",
        f"| KV case, the PRs' values: first token | {median_range(kv_pr)} | {f['kv_pr']} | {BOX['kv_pr_ms']} |",
        f"| Prefill case, stock: the reply's token gap (1st to 16th) | {median_range(pf_stock)} | "
        f"{f['prefill_stock']} | {BOX['prefill_stock_gap_ms']} |",
        f"| Prefill case, the PRs' values: the token gap | {median_range(pf_pr)} | {f['prefill_pr']} "
        f"| {BOX['prefill_pr_gap_ms']} |",
    ]
    # The time-equivalent caps' runs, by cap and threshold (stock1536-768-tail-N); vLLM's default is stock2048-.
    caps = sorted(
        (int(cap), int(threshold))
        for p in (runs / "block2").glob("stock*-*-tail-1.txt")
        if (parts := p.stem.removeprefix("stock").split("-"))[0].isdigit()
        and len(parts) == 4
        and (cap := parts[0]).isdigit()
        and (threshold := parts[1]).isdigit()
    )
    tails = []
    for label, block, prefix in (
        ("frozen, 512/256", "block1", "frozen-tail"),
        ("stock, 2,048/0", "block2", "stock2048-tail"),
        *(
            (f"stock, {cap:,}/{threshold:,}", "block2", f"stock{cap}-{threshold}-tail")
            for cap, threshold in caps
        ),
        ("the PRs' values", "block3", "pr-tail"),
    ):
        p95 = [
            r.ttft["p95"][0]
            for r in results.runs(runs, block, prefix).values()
            if isinstance(r, results.Tail) and r.ttft
        ]
        if p95:
            tails.append(f"{label} {median_range(p95)}")
    if tails:
        lines += [
            "",
            "The calls' first-token p95 beside the agents (tail.py): " + "; ".join(tails) + ".",
        ]
    return lines


def running_shares(points: list[Path]) -> dict[str, list[float]]:
    """Each configuration's measured running shares (the points' summary.json kv), by arm label."""
    shares: dict[str, list[float]] = {}
    for point in points:
        try:
            run = json.loads((point / "run.json").read_text())
            summary = json.loads((point / "summary.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        kv = summary.get("kv") or {}
        if "running_share" in kv:
            shares.setdefault(run["arm"]["label"], []).append(kv["running_share"])
    return shares


def report(host: Path, corpus: Path, binary: Path, out: Path, lh_bench: Path | None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    sim = calibration.Sim(binary, corpus, out)
    runs = host / "runs" / "a100"
    gpu = gpu_name(host)
    forecast = read_forecast(forecast_for(gpu))
    core = forecast.core
    lines = [
        f"# The {forecast.card} run against its forecast",
        "",
        f"From {host}. Every A100 number here is measured on the A100 unless it says modeled (the forecast, "
        f"{_shown(forecast.path)}) or names the box. {forecast_line(gpu, forecast)}",
        "",
        "## 1. The step model",
        "",
    ]
    export, e0 = runs / "calibration-s16", runs / "block1" / "frozen-e0" / "summary.json"
    steps_path = None
    measured_point = (export / "after.prom").exists() and e0.exists()
    if measured_point and not binary.exists():
        lines.append(
            f"Not computed: no lh-sim at {binary} (cargo build --release --locked -p lh-sim)."
        )
    elif measured_point:
        log = out / "calibration-s16-replay-log.csv"
        sim.replay_log("calibration-s16-replay-log", export)
        steps, fitted = fit(export, log, e0, out / "steps.json")
        steps_path = out / "steps.json"
        lines += step_model_lines(steps, forecast.costs)
        lines += ["", "The fit's own account:", "", "```", fitted.strip(), "```"]
    else:
        lines.append(
            "Not measured: no calibration point (runs/a100/calibration-s16) or no E0 (block 1's)."
        )
    lines += ["", "## 2. The calibration table", ""]
    if steps_path is not None:
        rows, _ = calibration_lines(
            sim, export, 16, ["--costs", str(e0), "--step-costs", str(steps_path)], "a100-s16"
        )
        lines += [*TABLE_HEAD, *rows]
    else:
        lines.append("Not computed." if measured_point else "Not measured.")
    lines += ["", "## 3. The probes", ""]
    if any((runs / block).is_dir() for block in ("block1", "block2", "block3")):
        lines += probe_lines(runs, forecast)
    else:
        lines.append("Not measured: no probe block (runs/a100/block1-3).")
    lines += ["", "## 4. E2 within the A100", ""]
    # The core's rounds alone (a100-r1 to -r5): the optional arms' campaigns begin a100- too (a100-resubmit-r3 ...).
    points = sorted(p.parent for p in host.glob("results/e2/a100-r[0-9]*/*/run.json"))
    if points:
        shares = running_shares(points)
        if lh_bench is not None:
            subprocess.run(
                [str(lh_bench), "e2", "report", *map(str, points), "--out", str(out / "e2")],
                check=True,
                capture_output=True,
            )
            lines += [(out / "e2" / "frontier.md").read_text().strip(), ""]
        lines += [
            "The measured running share (the time free KV was under two calls' 2,048 tokens while agents ran), "
            f"against the forecast's {core['running']} (central) for the core [modeled]:",
            "",
        ]
        lines += [
            f"- {label}: {', '.join(f'{100 * s:.0f}%' for s in values)}"
            for label, values in shares.items()
        ] or ["- Not measured: no point's summary.json has a running share (kv.running_share)."]
    else:
        lines.append("Not measured: no E2 point (results/e2/a100-r1 to -r5).")
    lines += [
        "",
        "## 5. Across cards (different hardware)",
        "",
        "The core scenario's regime coordinates: the box measured (E2 W5, docs/BENCHMARKS.md) a saturated card, 99% "
        f"of agent requests waiting for blocks and a running share of 17%; the forecast puts the {forecast.card} at "
        f"{core['busy']} busy, {core['stalled']} stalled and {core['running']} [modeled]. Each card's distance to its "
        "own SLO (its voice alone + 250 ms) is in §4 for the A100 and in docs/BENCHMARKS.md for the box; the two are "
        "compared in these coordinates only.",
    ]
    path = out / "a100.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def calibration_check(exports: Path, corpus: Path, binary: Path, out: Path) -> list[str]:
    """report's calibration code on the box's W2 points with the box's E0, against calibration.py report's rows:
    the rows that differ (none when it reproduces them)."""
    out.mkdir(parents=True, exist_ok=True)
    expected = calibration.report(exports, corpus, binary, out / "box").splitlines()
    sim = calibration.Sim(binary, corpus, out)
    pairs = []
    for _, _, run in calibration.POINTS:
        sim.replay_log(f"{run}-replay-log", exports / run)
        pairs.append((exports / run / calibration.FILES[0], out / f"{run}-replay-log.csv"))
    steps, _ = calibration.fit(pairs, out / "steps.json")
    differ = []
    for label, sessions, run in calibration.POINTS:
        rows, _ = calibration_lines(sim, exports / run, sessions, ["--step-costs", str(steps)], run)
        ours = [rows[0].replace(f"| {sessions} sessions |", f"| {label} |"), *rows[1:]]
        for row in ours:
            if row not in expected:
                differ.append(row)
    return differ


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="analysis.py", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("report", "calibration-check"):
        p = commands.add_parser(name)
        p.add_argument(
            "dir", type=Path, help="the host's records (report) or the box's exports (check)"
        )
        p.add_argument("--corpus", type=Path, required=True, help="the whole 32x corpus, weka-x32")
        p.add_argument("--lh-sim", type=Path, required=True, help="lh-sim's release binary")
        p.add_argument("--out", type=Path, required=True)
        if name == "report":
            p.add_argument(
                "--lh-bench", type=Path, help="lh-bench, for E2's report (the bench venv's)"
            )
    args = parser.parse_args(argv)
    if args.command == "report":
        print(report(args.dir, args.corpus, args.lh_sim, args.out, args.lh_bench))
        return 0
    differ = calibration_check(args.dir, args.corpus, args.lh_sim, args.out)
    for row in differ:
        print(f"analysis: not in calibration.py's table: {row}", file=sys.stderr)
    print(
        f"analysis: {'reproduces' if not differ else 'does not reproduce'} the box's calibration rows"
    )
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
