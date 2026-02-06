"""
plotting.py
- Loads a run folder from ./runs/<RUN_ID>/
- Computes derived quantities
- Writes summary CSVs and plots into runs/<RUN_ID>/summary/
"""

import argparse
from pathlib import Path

from load_run import load_all_shots
from features import compute_summary_tables
from plots import make_all_plots

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Path to a local run folder under runs/")
    args = ap.parse_args()

    run_dir = Path(args.run)
    shots = load_all_shots(run_dir)

    summary_dir = run_dir / "summary"
    summary_dir.mkdir(exist_ok=True)

    tables = compute_summary_tables(shots)
    # TODO: write tables to CSV(s)

    make_all_plots(shots, outdir=summary_dir / "plots")
    print(f"Analysis done: {summary_dir}")

if __name__ == "__main__":
    main()
