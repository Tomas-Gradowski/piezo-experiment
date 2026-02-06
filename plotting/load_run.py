"""
Load raw shot CSVs from runs/<RUN_ID>/shots/
Return a list/dict of shot objects with metadata.
"""

from pathlib import Path
import pandas as pd

def load_all_shots(run_dir: Path):
    shots_dir = run_dir / "shots"
    if not shots_dir.exists():
        raise FileNotFoundError(f"No shots/ found in {run_dir}")

    shots = []
    for csv_path in sorted(shots_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        shots.append({"path": csv_path, "df": df})
    return shots
