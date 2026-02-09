#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal, List, Dict, Any

Scale = Literal["linear", "log"]


# -----------------------------
# Helpers
# -----------------------------
#
def nearest_power_of_two(x: float) -> int:
    """Closest power-of-two integer to x, clamped to >=1."""
    if x <= 1:
        return 1
    return int(max(1, 2 ** round(math.log(x, 2))))


def _round_like_csharp(tmp: float) -> float:
    """
    Mimic the C# rounding:
      order = int(log10(tmp)+1)
      round(round(tmp/10^order, 3) * 10^order)
    Keeps ~3 significant digits-ish.
    """
    if tmp == 0:
        return 0.0
    order = int(math.floor(math.log10(abs(tmp))) + 1)
    scaled = tmp / (10 ** order)
    return round(round(scaled, 3) * (10 ** order))


def generate_array(min_v: float, max_v: float, n: int, scale: Scale) -> List[float]:
    if n <= 0:
        raise ValueError("number of points must be >= 1")
    if n == 1:
        return [float(min_v)]

    if scale == "linear":
        delta = (max_v - min_v) / (n - 1)
        return [_round_like_csharp(min_v + i * delta) for i in range(n)]

    if scale == "log":
        if min_v <= 0 or max_v <= 0:
            raise ValueError("log scale requires min and max > 0")
        ratio = 10 ** ((math.log10(max_v) - math.log10(min_v)) / (n - 1))
        vals = [float(min_v)]
        for _ in range(1, n):
            vals.append(vals[-1] * ratio)
        return [_round_like_csharp(v) for v in vals]

    raise ValueError(f"Unknown scale: {scale}")


def effective_resistance_total(load_r: float, osc_r: float) -> float:
    """Same idea as C#: resistancesTotal = 1 / (1/R + 1/osc_r)."""
    return 1.0 / (1.0 / load_r + 1.0 / osc_r)


# -----------------------------
# Config models
# -----------------------------

@dataclass
class PitayaConfig:
    host: str
    port: int
    fs_hz: float
    n_samples: int
    trig_delay: int
    units: str
    gain_ch1: str
    gain_ch2: str


@dataclass
class ExperimentConfig:
    # meta
    sample_name: str
    created_at: str
    output_dir: str

    # sweeps
    min_freq_hz: float
    max_freq_hz: float
    freq_points: int
    freq_scale: Scale

    min_r_ohm: float
    max_r_ohm: float
    r_points: int
    r_scale: Scale

    repetitions: int
    periods: float

    # calibration / model params
    pressure_scale: float
    osc_r_ohm: float
    max_decimation: int

    pitaya: PitayaConfig


# -----------------------------
# Plan building
# -----------------------------

def compute_decimation(cfg: ExperimentConfig, freq_hz: float) -> int:
    # C# intent: nearestPowerOf2( (1/f)*periods*Fs / 16384 ), clamped to 2^16
    target = (1.0 / freq_hz) * cfg.periods * cfg.pitaya.fs_hz / cfg.pitaya.n_samples
    dec = nearest_power_of_two(target)
    return int(min(dec, cfg.max_decimation))


def pitaya_setup_commands(cfg: ExperimentConfig, decimation: int) -> List[str]:
    p = cfg.pitaya
    return [
        "ACQ:RST",
        f"ACQ:DATA:UNITS {p.units}",
        f"ACQ:SOUR1:GAIN {p.gain_ch1}",
        f"ACQ:SOUR2:GAIN {p.gain_ch2}",
        f"ACQ:DEC {decimation}",
        f"ACQ:TRIG:DLY {p.trig_delay}",
        "ACQ:START",
        "ACQ:TRIG NOW",
    ]


def build_plan(cfg: ExperimentConfig) -> Dict[str, Any]:
    freqs = generate_array(cfg.min_freq_hz, cfg.max_freq_hz, cfg.freq_points, cfg.freq_scale)
    rs = generate_array(cfg.min_r_ohm, cfg.max_r_ohm, cfg.r_points, cfg.r_scale)
    rs_total = [effective_resistance_total(r, cfg.osc_r_ohm) for r in rs]

    points = []
    idx = 0
    for r_idx, r in enumerate(rs):
        for f_idx, f in enumerate(freqs):
            dec = compute_decimation(cfg, f)
            points.append({
                "index": idx,
                "r_index": r_idx,
                "f_index": f_idx,
                "freq_hz": f,
                "load_r_ohm": r,
                "total_r_ohm": rs_total[r_idx],
                "decimation": dec,
                "pitaya_setup": pitaya_setup_commands(cfg, dec),
                "pitaya_queries": [
                    "ACQ:SOUR1:DATA?",
                    "ACQ:SOUR2:DATA?",
                ],
            })
            idx += 1

    return {
        "frequencies_hz": freqs,
        "resistances_ohm": rs,
        "resistances_total_ohm": rs_total,
        "grid_points": points,
    }


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate experiment config + per-point Pitaya command plan (old UI parameters via CLI)."
    )

    # Meta / output
    ap.add_argument("--sample-name", required=True, help="Equivalent to textBoxSampleName")
    ap.add_argument("--out-dir", default="generated_configs", help="Base output folder (equiv. textBoxFolder)")
    ap.add_argument("--run-subdir", default="auto",
                    help="Subfolder name. Use 'auto' to mimic date-based folder naming.")

    # Frequency sweep
    ap.add_argument("--min-freq", type=float, required=True, help="Minimum frequency in Hz (textBoxMinFreq)")
    ap.add_argument("--max-freq", type=float, required=True, help="Maximum frequency in Hz (textBoxMaxFreq)")
    ap.add_argument("--freq-points", type=int, required=True, help="Number of frequency points (textBoxFreqPoints)")
    ap.add_argument("--freq-scale", choices=["linear", "log"], required=True, help="dropBoxFreq")

    # Resistance sweep
    ap.add_argument("--min-r", type=float, required=True, help="Minimum resistance in ohms (textBoxMinR)")
    ap.add_argument("--max-r", type=float, required=True, help="Maximum resistance in ohms (textBoxMaxR)")
    ap.add_argument("--r-points", type=int, required=True, help="Number of resistance points (textBoxRPoints)")
    ap.add_argument("--r-scale", choices=["linear", "log"], required=True, help="dropBoxR")

    # Measurement params
    ap.add_argument("--repetitions", type=int, default=1, help="textBoxRepetitions")
    ap.add_argument("--periods", type=float, default=10, help="textBoxPeriods (how many periods to capture)")

    # Pitaya params (defaults match your code)
    ap.add_argument("--pitaya-host", default="rp-f06549.local")
    ap.add_argument("--pitaya-port", type=int, default=5000)
    ap.add_argument("--fs-hz", type=float, default=125e6)
    ap.add_argument("--n-samples", type=int, default=16384)
    ap.add_argument("--trig-delay", type=int, default=1 * (2 ** 13))
    ap.add_argument("--units", default="VOLTS")
    ap.add_argument("--gain-ch1", default="HV")
    ap.add_argument("--gain-ch2", default="HV")

    # Analysis / calibration params
    ap.add_argument("--pressure-scale", type=float, default=0.5, help="Pressure multiplier (your 0.5)")
    ap.add_argument("--osc-r-ohm", type=float, default=1e6, help="Oscilloscope/input equivalent R (used in R_total)")
    ap.add_argument("--max-decimation", type=int, default=2 ** 16)

    return ap.parse_args()


def make_output_path(base: str, run_subdir: str, sample_name: str) -> Path:
    base_path = Path(base)

    if run_subdir == "auto":
        # A simpler, stable version of your C# folder nesting.
        # (If you want the exact Year/MonthName/etc structure, we can copy it exactly too.)
        stamp = datetime.now().strftime("%Y_%m_%d_%Hh%M")
        return base_path / stamp / sample_name

    return base_path / run_subdir / sample_name


def main() -> None:
    args = parse_args()

    base_dir = Path(args.out_dir).expanduser()

    # optional: fail early with a clear error if base is unwritable
    base_dir.mkdir(parents=True, exist_ok=True)
    if not base_dir.is_dir():
        raise RuntimeError(f"--out-dir is not a directory: {base_dir}")

    out_path = make_output_path(str(base_dir), args.run_subdir, args.sample_name)
    out_path.mkdir(parents=True, exist_ok=True)
 
    pitaya_cfg = PitayaConfig(
        host=args.pitaya_host,
        port=args.pitaya_port,
        fs_hz=args.fs_hz,
        n_samples=args.n_samples,
        trig_delay=args.trig_delay,
        units=args.units,
        gain_ch1=args.gain_ch1,
        gain_ch2=args.gain_ch2,
    )

    cfg = ExperimentConfig(
        sample_name=args.sample_name,
        created_at=datetime.now().isoformat(timespec="seconds"),
        output_dir=str(out_path),

        min_freq_hz=args.min_freq,
        max_freq_hz=args.max_freq,
        freq_points=args.freq_points,
        freq_scale=args.freq_scale,

        min_r_ohm=args.min_r,
        max_r_ohm=args.max_r,
        r_points=args.r_points,
        r_scale=args.r_scale,

        repetitions=args.repetitions,
        periods=args.periods,

        pressure_scale=args.pressure_scale,
        osc_r_ohm=args.osc_r_ohm,
        max_decimation=args.max_decimation,

        pitaya=pitaya_cfg,
    )

    plan = build_plan(cfg)

    # Write outputs
    (out_path / "experiment_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    (out_path / "measurement_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    # Convenience: template of the first point
    first = plan["grid_points"][0]
    (out_path / "pitaya_setup_first_point.txt").write_text(
        "\n".join(first["pitaya_setup"] + first["pitaya_queries"]) + "\n",
        encoding="utf-8"
    )

    print(f"Generated in: {out_path}")
    print(f"- experiment_config.json")
    print(f"- measurement_plan.json")
    print(f"- pitaya_setup_first_point.txt")


if __name__ == "__main__":
    main()
