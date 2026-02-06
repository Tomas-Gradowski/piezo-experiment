#!/usr/bin/env python3
"""
Dummy acquisition pipeline (no hardware).

Creates:
runs/<RUN_ID>/
  config.yaml
  run_metadata.json
  shots/shot_0001_R10000_f10_rep0.csv
  ...
"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class ShotMeta:
    shot_id: int
    R_ohm: float
    f_hz: float
    rep: int


def make_run_id(sample_name: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in sample_name)
    return f"{ts}_{safe}"


def generate_dummy_waveform(f_hz: float, duration_s: float, fs_hz: float, R_ohm: float, seed: int) -> pd.DataFrame:
    """
    Produce synthetic pressure + voltage signals with noise.
    - pressure ~ sin(2π f t) + offset
    - voltage amplitude depends loosely on R to mimic loading effects
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs_hz)
    t = np.arange(n) / fs_hz

    # Pressure: 0.2 bar amplitude at low f, a bit higher at high f (arbitrary)
    p_amp = 0.15 + 0.002 * f_hz
    p_off = 1.0
    pressure = p_off + p_amp * np.sin(2 * np.pi * f_hz * t + 0.3) + rng.normal(0, 0.01, size=n)

    # Voltage: pretend there is an optimal load region (totally arbitrary)
    # We'll create a soft peak vs log(R)
    logR = np.log10(max(R_ohm, 1.0))
    v_amp = 2.0 * np.exp(-0.5 * ((logR - 4.5) / 0.7) ** 2)  # peak near ~30k-100k
    voltage = v_amp * np.sin(2 * np.pi * f_hz * t + 1.1) + rng.normal(0, 0.02, size=n)

    return pd.DataFrame({"t_s": t, "pressure_bar": pressure, "voltage_v": voltage})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/example.yaml")
    ap.add_argument("--out", default="runs", help="Local runs directory")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    sample_name = cfg["run"]["sample_name"]
    periods = int(cfg["run"]["periods"])
    reps = int(cfg["run"]["repetitions"])
    resistances = list(cfg["sweep"]["resistances_ohm"])
    freqs = list(cfg["sweep"]["frequencies_hz"])

    run_id = make_run_id(sample_name)
    run_dir = Path(args.out) / run_id
    shots_dir = run_dir / "shots"
    shots_dir.mkdir(parents=True, exist_ok=True)

    # Save config used
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # Metadata
    meta = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "dummy",
        "columns": ["t_s", "pressure_bar", "voltage_v"],
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    # Generate shots
    shot_id = 1
    for R in resistances:
        for f in freqs:
            duration_s = periods / float(f)
            fs_hz = 5000.0  # arbitrary dummy sampling rate
            for rep in range(reps):
                sm = ShotMeta(shot_id=shot_id, R_ohm=float(R), f_hz=float(f), rep=rep)

                df = generate_dummy_waveform(
                    f_hz=sm.f_hz,
                    duration_s=duration_s,
                    fs_hz=fs_hz,
                    R_ohm=sm.R_ohm,
                    seed=1000 + shot_id,
                )

                fname = f"shot_{shot_id:04d}_R{int(sm.R_ohm)}_f{int(sm.f_hz)}_rep{sm.rep}.csv"
                df.to_csv(shots_dir / fname, index=False)
                shot_id += 1

    print(f"✅ Dummy run created: {run_dir}")


if __name__ == "__main__":
    main()
