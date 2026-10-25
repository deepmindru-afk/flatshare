"""benchmarks/a100/analysis.py on synthetic records: the forecast's costs read back, each fitted term set beside them,
the probes beside the forecast and the box, the running shares by configuration, and a report that says what the
records lack. (Its calibration code runs on the box's W2 exports in `make check-a100-calibration`, and the whole
report on the CPU dry run's records in test_a100_dry_run.py.)"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
A100 = REPO / "benchmarks" / "a100"


def _module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analysis = _module("lh_a100_analysis", A100 / "analysis.py")
DASH = "\u2013"  # the range dash
PROBE = (
    "{case}: priority-0 TTFT {ttft} ms, 16 tokens {t16} ms, for 761 prompt tokens, sent with 2 priority-2 requests of"
    " 17999 tokens running and 24 of 2274 blocks free; after it, 2 running and 0 waiting\n"
)
TAIL = (
    "calls: 40 turns, 0 failed (counted at 30,000 ms), TTFT p50 110 p90 300 p95 {p95} max 900 ms; agents: 0 of 12"
    " stopped early\n"
)


def test_the_forecasts_costs_read_back() -> None:
    costs = analysis.forecast_costs(analysis.FORECAST.read_text())
    assert costs["central"] == {
        "step_ms": 4.55,
        "us_per_token": 41.33,
        "ns_per_pair": 2.59,
        "ns_per_context": 123.7,
    }
    assert costs["low"]["us_per_token"] == 29.19 and costs["high"]["ns_per_pair"] == 3.78
    with pytest.raises(ValueError):
        analysis.forecast_costs("## Step and tier costs, Qwen3-4B-Instruct-2507-FP8\n\nno table\n")


def test_each_fitted_term_stands_beside_its_forecast() -> None:
    forecast = analysis.forecast_costs(analysis.FORECAST.read_text())
    steps = {"step_ms": 5.0, "us_per_token": 60.0, "ns_per_pair": 2.6, "ns_per_context": 120.0}
    lines = analysis.step_model_lines(steps, forecast)
    assert (
        f"| Step: fixed | 5 ms | 4.55 ms | 3.33{DASH}4.55 ms | 1.10, outside the range |" in lines
    )
    assert f"| per pair | 2.6 ns | 2.59 ns | 2.15{DASH}3.78 ns | 1.00 |" in lines
    # 60 µs a token is 1.45 times the forecast: the largest miss, and what it points at.
    assert lines[-1].startswith(
        "The largest miss is the per token term, 1.45 times the forecast (outside its range)"
    )
    assert "Marlin" in lines[-1]


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def probe_runs(runs: Path) -> None:
    for i, (ttft, t16) in enumerate(((1600, 1700), (1700, 1800), (1650, 1750)), 1):
        write(runs / "block1" / f"frozen-kv-{i}.txt", PROBE.format(case="kv", ttft=ttft, t16=t16))
    for i, ttft in enumerate((140, 160, 150), 1):
        write(
            runs / "block3" / f"pr-kv-{i}.txt", PROBE.format(case="kv", ttft=ttft, t16=ttft + 120)
        )
    for i, (ttft, t16) in enumerate(((300, 800), (320, 830)), 1):  # 33.3 and 34.0 ms a token
        write(
            runs / "block1" / f"frozen-prefill-{i}.txt",
            PROBE.format(case="prefill", ttft=ttft, t16=t16),
        )
    write(
        runs / "block3" / "pr-prefill-1.txt", PROBE.format(case="prefill", ttft=200, t16=290)
    )  # 6.0 ms
    for prefix, block, p95s in (
        ("frozen-tail", "block1", (1800, 1900)),
        ("stock2048-tail", "block2", (1500, 1700)),
        ("stock1536-tail", "block2", (1400, 1600)),
        ("pr-tail", "block3", (300, 340)),
    ):
        for i, p95 in enumerate(p95s, 1):
            write(runs / block / f"{prefix}-{i}.txt", TAIL.format(p95=p95))


def test_the_probes_stand_beside_the_forecast_and_the_box(tmp_path: Path) -> None:
    probe_runs(tmp_path)
    lines = analysis.probe_lines(tmp_path)
    text = "\n".join(lines)
    assert (
        f"| KV case, stock: the voice turn's wait | 1,650 ms (1,600{DASH}1,700, 3 runs) | 1,636{DASH}1,688 ms"
        in text
    )
    assert (
        f"| KV case, the PRs' values: first token | 150 ms (140{DASH}160, 3 runs) | 134{DASH}158 ms"
        in text
    )
    assert (
        f"| Prefill case, stock: the reply's token gap (1st to 16th) | 34 ms (33{DASH}34, 2 runs) |"
        in text
    )
    assert "| Prefill case, the PRs' values: the token gap | 6 ms | 6.6 ms" in text
    assert "| KV case, stock at 2,048/0 | not measured |" in text
    assert (
        "The calls' first-token p95 beside the agents (tail.py): frozen, 512/256 1,850 ms "
        f"(1,800{DASH}1,900, 2 runs); stock, 2,048/0 1,600 ms (1,500{DASH}1,700, 2 runs); stock, 1,536/768 1,500 ms "
        f"(1,400{DASH}1,600, 2 runs); the PRs' values 320 ms (300{DASH}340, 2 runs)."
    ) in text


def test_running_shares_by_configuration(tmp_path: Path) -> None:
    for name, label, share in (
        ("a", "frozen", 0.12),
        ("b", "frozen", 0.15),
        ("c", "PRs", 0.05),
        ("d", "voice", None),
    ):
        write(tmp_path / name / "run.json", json.dumps({"arm": {"label": label}}))
        kv = {"running_share": share} if share is not None else None
        write(tmp_path / name / "summary.json", json.dumps({"kv": kv}))
    points = sorted(tmp_path.iterdir())
    assert analysis.running_shares(points) == {"frozen": [0.12, 0.15], "PRs": [0.05]}


def test_a_report_says_what_its_records_lack(tmp_path: Path) -> None:
    host = tmp_path / "host"
    probe_runs(host / "runs" / "a100")
    out = tmp_path / "out"
    # No calibration point and no E2 point: lh-sim and lh-bench are never called.
    path = analysis.report(host, tmp_path / "corpus", tmp_path / "no-lh-sim", out, None)
    text = path.read_text()
    assert "## 1. The step model\n\nNot measured: no calibration point" in text
    assert "## 2. The calibration table\n\nNot measured." in text
    assert "| block1/frozen-kv-1 | kv | 1600 | 1700 |" in text
    assert "## 4. E2 within the A100\n\nNot measured: no E2 point" in text
    assert "## 5. Across cards (different hardware)" in text
