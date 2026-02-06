"""
testing.py
- Reads config
- For each resistance (manual decade box): prompts user to set it and confirms
- Runs acquisition for each frequency/repetition
- Saves temporary files on Red Pitaya (remote)
- Pulls the run folder back via scp into ./runs/<RUN_ID>/

Note: saving on RP is optional; you can also stream directly to the PC later.
"""

import argparse
from pathlib import Path
import yaml

from run_layout import make_run_id, local_run_dir, remote_run_dir
from rp_scpi import RedPitayaSCPI
from rp_scp import scp_pull_dir, ssh_mkdir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    host = cfg["redpitaya"]["host"]
    port = cfg["redpitaya"]["scpi_port"]
    user = cfg["redpitaya"]["user"]
    remote_base = cfg["redpitaya"]["remote_runs_dir"]

    run_id = make_run_id(cfg["run"]["sample_name"])
    local_dir = local_run_dir(cfg["output"]["local_runs_dir"], run_id)
    remote_dir = remote_run_dir(remote_base, run_id)

    local_dir.mkdir(parents=True, exist_ok=True)

    # Ensure remote directory exists (SSH)
    ssh_mkdir(user=user, host=host, remote_path=str(remote_dir))

    rp = RedPitayaSCPI(host=host, port=port)
    rp.connect()

    # TODO: write run_metadata locally (config copy, timestamps, etc.)

    for R in cfg["sweep"]["resistances_ohm"]:
        input(f"\nSet DECADE BOX to R = {R} Ω, then press ENTER to continue...")

        for f in cfg["sweep"]["frequencies_hz"]:
            for rep in range(cfg["run"]["repetitions"]):
                # TODO:
                # 1) configure acquisition (decimation, channels, trigger)
                # 2) start acquisition
                # 3) read CH1/CH2 data
                # 4) write a CSV on the RP under remote_dir/shots/
                # For now, just a placeholder:
                print(f"Would measure: R={R}Ω  f={f}Hz  rep={rep}")

    rp.close()

    # Pull remote run back to local runs/
    scp_pull_dir(user=user, host=host, remote_path=str(remote_dir), local_path=str(local_dir))

    print(f"\nDone. Local run saved to: {local_dir}")

if __name__ == "__main__":
    main()
