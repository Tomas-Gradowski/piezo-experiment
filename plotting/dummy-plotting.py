#!/usr/bin/env python3
"""
Offline analysis for a run folder.

- Loads all shots in runs/<RUN_ID>/shots/*.csv
- Computes quick metrics:
  - pressure amplitude (peak-to-peak / 2)
  - Vrms
- Writes:
  - runs/<RUN_ID>/summary/summary.csv
  - runs/<RUN_ID>/summary/plots/...
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SHOT_RE = re.compile(r"shot_(\d+)_R(\d+)_f(\d+)_rep(\d+)\.csv")


def parse_meta_from_name(path: Path) -> dict:
    m = SHOT_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected shot filename: {path.name}")
    shot_id, R, f, rep = m.groups()
    return {"shot_id": int(shot_id), "R_ohm": float(R), "f_hz": float(f), "rep": int(rep)}


def vrms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Path to runs/<RUN_ID>")
    args = ap.parse_args()

    run_dir = Path(args.run)
    shots_dir = run_dir / "shots"
    if not shots_dir.exists():
        raise FileNotFoundError(f"No shots directory: {shots_dir}")

    out_summary = run_dir / "summary"
    out_plots = out_summary / "plots"
    out_plots.mkdir(parents=True, exist_ok=True)

    rows = []
    for csv_path in sorted(shots_dir.glob("*.csv")):
        meta = parse_meta_from_name(csv_path)
        df = pd.read_csv(csv_path)

        p = df["pressure_bar"].to_numpy()
        v = df["voltage_v"].to_numpy()

        p_amp = 0.5 * (float(np.max(p) - np.min(p)))  # simple estimate
        v_rms = vrms(v)

        rows.append({**meta, "pressure_amp_bar": p_amp, "v_rms_v": v_rms})

    summary = pd.DataFrame(rows).sort_values(["R_ohm", "f_hz", "rep"])
    out_summary.mkdir(exist_ok=True)
    summary_path = out_summary / "summary.csv"
    summary.to_csv(summary_path, index=False)

    # Plot: Vrms vs R for each f (averaged over reps)
    g = summary.groupby(["R_ohm", "f_hz"], as_index=False).agg(
        v_rms_v=("v_rms_v", "mean"),
        pressure_amp_bar=("pressure_amp_bar", "mean"),
    )

    for f_hz, df_f in g.groupby("f_hz"):
        plt.figure()
        plt.semilogx(df_f["R_ohm"], df_f["v_rms_v"], marker="o")
        plt.xlabel("Load Resistance R (Ohm)")
        plt.ylabel("Voltage Vrms (V)")
        plt.title(f"Vrms vs R (f={int(f_hz)} Hz)")
        plt.grid(True, which="both")
        plt.tight_layout()
        plt.savefig(out_plots / f"vrms_vs_R_f{int(f_hz)}.png")
        plt.close()

    # Plot: pressure amplitude vs R for each f
    for f_hz, df_f in g.groupby("f_hz"):
        plt.figure()
        plt.semilogx(df_f["R_ohm"], df_f["pressure_amp_bar"], marker="o")
        plt.xlabel("Load Resistance R (Ohm)")
        plt.ylabel("Pressure amplitude (bar)")
        plt.title(f"Pressure amplitude vs R (f={int(f_hz)} Hz)")
        plt.grid(True, which="both")
        plt.tight_layout()
        plt.savefig(out_plots / f"pressure_amp_vs_R_f{int(f_hz)}.png")
        plt.close()

    print(f"✅ Wrote {summary_path}")
    print(f"✅ Plots in {out_plots}")


if __name__ == "__main__":
    main()
