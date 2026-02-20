# Piezo Experiment Control + Data Treatment

This repo contains a Python GUI and CLI tools to generate experiment plans, control Red Pitaya acquisition, drive an EPOS motor controller, and perform basic data treatment on the resulting CSV files.

Primary entry point: `ui6.py` (Tkinter GUI).

## Features
- Generate an experiment plan from frequency/resistance sweep parameters.
- Run acquisition on Red Pitaya (SCPI over TCP).
- Optional EPOS motor control with safety limits.
- Live plot preview (optional, requires matplotlib).
- Embedded data treatment to summarize amplitude metrics.

## Quick Start (GUI)
1. Create a Python virtual environment (recommended) and install dependencies.
2. Run the GUI.
3. Use the Run tab to select a mode and parameters.
4. Click `Generate + Run Selected Mode`.

Commands:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install matplotlib numpy pandas
python3 ui6.py
```

Notes:
- `matplotlib` is optional; without it the Live tab shows a message.
- `numpy` and `pandas` are required for the data treatment pipeline.
- `tkinter` is part of most Python distributions; on some Linux distros you may need `python3-tk` installed.

## Modes in the GUI
The UI is built around a single “Test Mode” selector. Internally it calls `gen_config.py` and `executor.py` with different flags.

- `plan_only`: Generate config, then run executor in `--dry-run` mode. No hardware connection.
- `pitaya_only`: Real Red Pitaya acquisition, no EPOS motion.
- `epos_only`: Real EPOS motion, no Red Pitaya acquisition.
- `full_experiment`: Both EPOS motion and Red Pitaya acquisition.

## CLI Workflow (No GUI)
You can run generation and execution manually.

1. Generate a run directory:
```bash
python3 gen_config.py \
  --sample-name TEST_SAMPLE \
  --min-freq 4 --max-freq 32 --freq-points 3 --freq-scale log \
  --min-r 1000 --max-r 10000 --r-points 2 --r-scale linear \
  --run-subdir auto --periods 10 --repetitions 1 \
  --pitaya-enabled --epos-enabled
```

2. Execute a run (real hardware):
```bash
python3 executor.py --run-dir generated_configs \
  --use-epos --rpm-limit 500 --prompt-rbox
```

3. Dry-run (no hardware):
```bash
python3 executor.py --run-dir generated_configs --dry-run --points 3
```

Notes:
- `--run-dir` can point to a specific run folder or to `generated_configs/`. The executor will find the most recent run.
- To disable Red Pitaya acquisition, add `--no-pitaya`.
- To jog the EPOS motor without running the plan, use `--epos-jog` (see `executor.py --help`).

## Data Treatment
There are two options:

1. Embedded: after a successful run, the GUI runs treatment automatically and writes a summary CSV.
2. Standalone script:
```bash
python3 Piezo_data-treatment_Python.py
```

The standalone script opens a small Tkinter UI to pick a directory and export a summary CSV.

## Output Structure
Generated content lives in `generated_configs/`:
- `generated_configs/<timestamp>/<sample_name>/experiment_config.json`
- `generated_configs/<timestamp>/<sample_name>/measurement_plan.json`
- `generated_configs/<timestamp>/<sample_name>/pitaya_setup_first_point.txt`
- `generated_configs/<timestamp>/<sample_name>/csv/*.csv`
- `generated_configs/<timestamp>/<sample_name>/run_manifest.json`

## Hardware / Network Setup
- Red Pitaya SCPI over TCP (default host `rp-f06549.local`, port `5000`).
- EPOS library path defaults to: `/home/tomas/thesis/epos/EPOS-Linux-Library-En/EPOS_Linux_Library/lib/intel/x86_64/libEposCmd.so.6.8.1.0`
- You can override Pitaya host/port via `executor.py` CLI flags or by editing the generated config.

Environment variables supported by the Red Pitaya client (`devices/pitaya_scpi.py`):
- `PITAYA_CONNECT_ATTEMPTS`
- `PITAYA_CONNECT_RETRY_DELAY_S`
- `PITAYA_HOST_FALLBACKS` (comma-separated host list)

## Troubleshooting
- If the GUI cannot find `gen_config.py` or `executor.py`, ensure you are running from the repo root.
- If `matplotlib` is not installed, the Live tab will be disabled.
- If EPOS fails to open, confirm the shared library path and USB device permissions.
- If Pitaya fails to connect, verify hostname/IP and that port `5000` is reachable.

## Making Changes
For a more detailed development guide, see `README_DEV.md`.

At a high level:
- UI logic and subprocess orchestration live in `ui6.py`.
- Plan generation logic is in `gen_config.py`.
- Execution and hardware control live in `executor.py` and `devices/`.
- The data treatment GUI is `Piezo_data-treatment_Python.py`.
