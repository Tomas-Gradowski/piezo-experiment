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

def freq_to_rpm(freq_hz: float, rpm_per_hz: float = 60.0) -> float:
    # if 1 Hz = 60 RPM (1 rotation per period), keep 60.0
    return freq_hz * rpm_per_hz


def dwell_time_s(freq_hz: float, periods: float, min_dwell_s: float = 0.25) -> float:
    # hold long enough to cover N periods, but never too tiny
    if freq_hz <= 0:
        raise ValueError("freq_hz must be > 0")
    return max(periods / freq_hz, min_dwell_s)


def generate_array(min_v: float, max_v: float, n: int, scale: Scale) -> List[float]:
    if n <= 0:
        raise ValueError("number of points must be >= 1")
    if n == 1:
        return [float(min_v)]

    if scale == "linear":
        delta = (max_v - min_v) / (n - 1)
        return [(min_v + i * delta) for i in range(n)]

    if scale == "log":
        if min_v <= 0 or max_v <= 0:
            raise ValueError("log scale requires min and max > 0")
        ratio = 10 ** ((math.log10(max_v) - math.log10(min_v)) / (n - 1))
        vals = [float(min_v)]
        for _ in range(1, n):
            vals.append(vals[-1] * ratio)
        return vals

    raise ValueError(f"Unknown scale: {scale}")


def effective_resistance_total(load_r: float, osc_r: float) -> float:
        return 1.0 / (1.0 / load_r + 1.0 / osc_r)

DR08_WEIGHTS_OHM = [1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1.0, 0.1]  # matches your UI boxes
DR08_WEIGHTS_TENTHS = [int(w * 10) for w in DR08_WEIGHTS_OHM]
DR08_MAX_TENTHS = 9 * sum(DR08_WEIGHTS_TENTHS)  # => 99_999_999 tenths = 9_999_999.9 ohm

def dr08_digits_from_ohms(r_ohm: float) -> tuple[list[int], float]:
    """
    Returns (digits[8], r_ohm_actual) where r_ohm_actual is quantized to 0.1Ω.
    """
    r10 = int(round(r_ohm * 10))
    if r10 < 0:
        raise ValueError("Resistance cannot be negative")
    if r10 > DR08_MAX_TENTHS:
        # choose ONE behavior: clamp or raise. I recommend clamping + recording actual.
        r10 = DR08_MAX_TENTHS

    digits = []
    rem = r10
    for w10 in DR08_WEIGHTS_TENTHS:
        d = rem // w10
        digits.append(int(d))
        rem -= int(d) * w10

    actual10 = sum(d * w for d, w in zip(digits, DR08_WEIGHTS_TENTHS))
    return digits, actual10 / 10.0
# -----------------------------
# Config models
# -----------------------------

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

    # motor execution parameters
    rpm_per_hz: float          # usually 60
    settle_s: float            # wait after changing speed before dwell
    min_dwell_s: float         # minimum dwell time per point

    # calibration / model params (keep if you still use them downstream)
    osc_r_ohm: float

# -----------------------------
# Plan building
# -----------------------------
def build_plan(cfg: ExperimentConfig) -> Dict[str, Any]:
    freqs = generate_array(cfg.min_freq_hz, cfg.max_freq_hz, cfg.freq_points, cfg.freq_scale)
    rs_target = generate_array(cfg.min_r_ohm, cfg.max_r_ohm, cfg.r_points, cfg.r_scale)

    resistance_steps = []
    flat_points = []
    idx = 0

    for r_idx, r_target in enumerate(rs_target):
        dr08_digits, r_actual = dr08_digits_from_ohms(r_target)
        r_total = effective_resistance_total(r_actual, cfg.osc_r_ohm)

        sweep_points = []
        for f_idx, f in enumerate(freqs):
            pt = {
                "index": idx,
                "r_index": r_idx,
                "f_index": f_idx,
                "freq_hz": f,
                "target_rpm": freq_to_rpm(f, cfg.rpm_per_hz),
                "settle_s": cfg.settle_s,
                "dwell_s": dwell_time_s(f, cfg.periods, cfg.min_dwell_s),

                "load_r_ohm_target": r_target,
                "load_r_ohm": r_actual,
                "total_r_ohm": r_total,
                "dr08_digits": dr08_digits,
            }
            sweep_points.append(pt)
            flat_points.append(pt)
            idx += 1

        resistance_steps.append({
            "r_index": r_idx,
            "load_r_ohm_target": r_target,
            "load_r_ohm": r_actual,
            "total_r_ohm": r_total,
            "dr08_digits": dr08_digits,
            "pre_action": {
                "type": "pause_for_user",
                "message": f"Set DR08 to {dr08_digits} (R={r_actual} Ω), then Resume."
            },
            "sweep_points": sweep_points,
        })

    return {
        "frequencies_hz": freqs,
        "resistances_ohm_target": rs_target,
        "resistance_steps": resistance_steps,
        "grid_points": flat_points,  # keep if your executor still expects a flat list
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

   
    # Analysis / calibration params
    ap.add_argument("--pressure-scale", type=float, default=0.5, help="Pressure multiplier (your 0.5)")
    ap.add_argument("--osc-r-ohm", type=float, default=1e6, help="Oscilloscope/input equivalent R (used in R_total)")
    ap.add_argument("--max-decimation", type=int, default=2 ** 16)

    return ap.parse_args()


def make_output_path(base: str, run_subdir: str, sample_name: str) -> Path:
    base_path = Path(base)

    if run_subdir == "auto":
       
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
    


if __name__ == "__main__":
    main()
