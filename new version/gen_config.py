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

def dr08_digits_from_ohms(r_ohm: float) -> tuple[list[int], float]:
    """
    Returns (digits[8], r_ohm_actual) where r_ohm_actual is quantized to 0.1Ω
    and representable by DR08 digits (0..9 each decade).
    """
    r10 = int(round(r_ohm * 10))
    if r10 < 0:
        raise ValueError("Resistance cannot be negative")
    if r10 > DR08_MAX_TENTHS:
        r10 = DR08_MAX_TENTHS  # clamp

    digits: list[int] = []
    rem = r10
    for w10 in DR08_WEIGHTS_TENTHS:
        d = min(9, rem // w10)
        digits.append(int(d))
        rem -= int(d) * w10

    actual10 = sum(d * w for d, w in zip(digits, DR08_WEIGHTS_TENTHS))
    return digits, actual10 / 10.0

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
DR08_WEIGHTS_OHM = [1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1.0, 0.1] # matches your UI boxes 
DR08_WEIGHTS_TENTHS = [int(w * 10) for w in DR08_WEIGHTS_OHM] 
DR08_MAX_TENTHS = 9 * sum(DR08_WEIGHTS_TENTHS) # => 99_999_999 tenths = 9_999_999.9 ohm 
def build_dr08_resistance_sweep(min_r_ohm: float,max_r_ohm: float,n: int,scale: Scale,) -> Dict[str, Any]:
    requested = generate_array(min_r_ohm, max_r_ohm, n, scale)

    entries = []
    for r in requested:
        digits, actual = dr08_digits_from_ohms(r)
        entries.append({
            "requested_ohm": float(r),
            "digits": digits,          # 8 digits [MΩ .. 0.1Ω decade]
            "rbox_ohm": float(actual), # actual achievable value
        })

    # dedupe by actual resistance (important on log sweeps)
    seen = set()
    unique = []
    for e in entries:
        key = int(round(e["rbox_ohm"] * 10))  # tenths as integer key
        if key in seen:
            continue
        seen.add(key)
        unique.append(e)

    # sort increasing by actual value
    unique.sort(key=lambda e: e["rbox_ohm"])

    return {
        "mode": "dr08",
        "requested_ohm": requested,
        "entries": unique,
        "unique_rbox_ohm": [e["rbox_ohm"] for e in unique],
    }

# -----------------------------
# Config models
# -----------------------------

@dataclass
class PitayaConfig:
    enabled: bool
    host: str
    port: int
    fs_hz: float
    n_samples: int
    trig_delay: int
    units: str
    gain_ch1: str
    gain_ch2: str
    signal_hz_multiplier: float
@dataclass
class EposPlan:
    enabled: bool
    lib_path: str
    node_id: int
    gear_reduction: int
    one_turn: int
    max_rpm: int  # motor-shaft RPM clamp at driver level
    accel: int
    decel: int
    direction: Literal["cw", "ccw"]
    alternate: bool
    rest_s: float
    rpm_override: int  # 0 means "derive from freq_hz"

@dataclass
class ExperimentConfig:
    # meta
    sample_name: str
    created_at: str
    output_dir: str

    # sweeps
    min_motor_hz: float
    max_motor_hz: float
    motor_points: int
    motor_scale: Scale

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
    epos:  EposPlan
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
    motor_hzs = generate_array(cfg.min_motor_hz, cfg.max_motor_hz, cfg.motor_points, cfg.motor_scale)
    rbox = build_dr08_resistance_sweep(cfg.min_r_ohm, cfg.max_r_ohm, cfg.r_points, cfg.r_scale)

    points = []
    idx = 0
    for r_idx, r_entry in enumerate(rbox["entries"]):
        r = r_entry["rbox_ohm"]
        digits = r_entry["digits"]

        for f_idx, motor_hz in enumerate(motor_hzs):
            # EPOS target
            if cfg.epos.rpm_override > 0:
                rpm_out = int(cfg.epos.rpm_override)
            else:
                rpm_out = int(round(motor_hz * 60.0))  # 1 rev per cycle assumption
            # Convert motor max RPM to an output-shaft cap using gear reduction.
            # This avoids mixing motor-RPM and output-RPM units.
            max_out_rpm_from_motor = max(1, int(cfg.epos.max_rpm) // max(1, int(cfg.epos.gear_reduction)))
            rpm_out = min(rpm_out, max_out_rpm_from_motor)

            # Pitaya sampling intent (NOT the motor knob directly)
            signal_hz = float(motor_hz) * float(cfg.pitaya.signal_hz_multiplier)
            dec = compute_decimation(cfg, signal_hz)

            points.append({
                "index": idx,
                "r_index": r_idx,
                "f_index": f_idx,

                "motor_hz": float(motor_hz),
                "rpm_out": int(rpm_out),
                "signal_hz": float(signal_hz),

                "rbox_ohm": r,
                "rbox_digits": digits,
                "requested_r_ohm": r_entry["requested_ohm"],

                "total_r_ohm": effective_resistance_total(r, cfg.osc_r_ohm),
                "decimation": dec,
                "pitaya_setup": pitaya_setup_commands(cfg, dec),
                "pitaya_queries": ["ACQ:SOUR1:DATA?", "ACQ:SOUR2:DATA?"],
            })
            idx += 1

    return {
        "motor_hz": motor_hzs,
        "rbox": rbox,
        "grid_points": points,
    }

# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate experiment config + per-point Pitaya command plan (old UI parameters via CLI)."
    )
    ap.add_argument("--signal_hz_multiplier", type=float, default=1.0, help="Pitaya expected signal_hz = motor_hz * multiplier (for decimation)")
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
    ap.add_argument("--pitaya-enabled", action="store_true") 
    ap.add_argument("--pitaya-host", default="rp-f06549.local")
    ap.add_argument("--pitaya-port", type=int, default=5000)
    ap.add_argument("--fs-hz", type=float, default=125e6)
    ap.add_argument("--n-samples", type=int, default=16384)
    ap.add_argument("--trig-delay", type=int, default=1 * (2 ** 13))
    ap.add_argument("--units", default="VOLTS")
    ap.add_argument("--gain-ch1", default="HV")
    ap.add_argument("--gain-ch2", default="HV")
    
    #EPOS parameters
    ap.add_argument("--epos-enabled", action="store_true")
    ap.add_argument("--epos-lib", default="/home/tomas/thesis/epos/EPOS-Linux-Library-En/EPOS_Linux_Library/lib/intel/x86_64/libEposCmd.so.6.8.1.0")
    ap.add_argument("--epos-node-id", type=int, default=1)
    ap.add_argument("--epos-gear", type=int, default=18)
    ap.add_argument("--epos-one-turn", type=int, default=73728, help="EPOS increments per mechanical output turn")
    ap.add_argument("--epos-max-rpm", type=int, default=5000)
    ap.add_argument("--epos-accel", type=int, default=250000, help="EPOS velocity profile acceleration")
    ap.add_argument("--epos-decel", type=int, default=250000, help="EPOS velocity profile deceleration")
    ap.add_argument("--epos-direction", choices=["cw", "ccw"], default="cw")
    ap.add_argument("--epos-alternate", action="store_true")
    ap.add_argument("--epos-rest-s", type=float, default=0.0)
    ap.add_argument("--epos-rpm-override", type=int, default=0, help="Output rpm. 0 => derive from freq_hz")
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
        enabled=bool(args.pitaya_enabled),
        host=args.pitaya_host,
        port=args.pitaya_port,
        fs_hz=args.fs_hz,
        n_samples=args.n_samples,
        trig_delay=args.trig_delay,
        units=args.units,
        gain_ch1=args.gain_ch1,
        gain_ch2=args.gain_ch2,
        signal_hz_multiplier=float(args.signal_hz_multiplier),
    )
    epos_plan = EposPlan(
        enabled=bool(args.epos_enabled),
        lib_path=args.epos_lib,
        node_id=args.epos_node_id,
        gear_reduction=args.epos_gear,
        one_turn=int(args.epos_one_turn),
        max_rpm=args.epos_max_rpm,
        accel=int(args.epos_accel),
        decel=int(args.epos_decel),
        direction=args.epos_direction,
        alternate=bool(args.epos_alternate),
        rest_s=float(args.epos_rest_s),
        rpm_override=int(args.epos_rpm_override),
)
    cfg = ExperimentConfig(
        sample_name=args.sample_name,
        created_at=datetime.now().isoformat(timespec="seconds"),
        output_dir=str(out_path),

        min_motor_hz=args.min_freq,
        max_motor_hz=args.max_freq,
        motor_points=args.freq_points,
        motor_scale=args.freq_scale,

        min_r_ohm=args.min_r,
        max_r_ohm=args.max_r,
        r_points=args.r_points,
        r_scale=args.r_scale,

        repetitions=args.repetitions,
        periods=args.periods,

        pressure_scale=args.pressure_scale,
        osc_r_ohm=args.osc_r_ohm,
        max_decimation=args.max_decimation,
        epos=epos_plan,
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
