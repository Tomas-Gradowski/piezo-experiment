"""
Plotting functions only.
No reading from hardware, no file naming logic.
"""

from pathlib import Path

def make_all_plots(shots, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    # TODO: voltage vs time, pressure vs time, summary plots
