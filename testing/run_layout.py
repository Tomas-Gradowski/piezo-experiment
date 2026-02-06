"""
Defines naming conventions for runs and file locations.
Keeps the filesystem layout consistent across testing + plotting.
"""

from pathlib import Path
from datetime import datetime

def make_run_id(sample_name: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in sample_name)
    return f"{ts}_{safe}"

def local_run_dir(local_runs_dir: str, run_id: str) -> Path:
    return Path(local_runs_dir) / run_id

def remote_run_dir(remote_runs_dir: str, run_id: str) -> Path:
    # remote path is still a path-like string; Path is fine for joining
    return Path(remote_runs_dir) / run_id
