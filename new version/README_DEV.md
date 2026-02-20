# Development Guide

This document explains the internal structure, configuration formats, and common change workflows.

## Repository Layout
- `ui6.py`: Tkinter GUI. Generates config, launches executor, streams logs, and runs embedded treatment.
- `gen_config.py`: Builds `experiment_config.json` and `measurement_plan.json` from CLI parameters.
- `executor.py`: Executes the plan, talks to Red Pitaya and EPOS, writes CSVs and a run manifest.
- `devices/pitaya_scpi.py`: Minimal SCPI TCP client for Red Pitaya.
- `devices/epos_driver.py`: EPOS4 wrapper (lazy-loads the shared library).
- `Piezo_data-treatment_Python.py`: Standalone data-treatment GUI.
- `generated_configs/`: Default output location for generated runs.
- `oldbroken ui/`: Legacy GUI versions (reference only).

## How the Pieces Fit Together
1. `ui6.py` assembles generator args and calls `gen_config.py` as a subprocess.
2. `gen_config.py` writes `experiment_config.json` (static configuration) and `measurement_plan.json` (per-point plan including pitaya commands).
3. `ui6.py` locates the newest run directory and launches `executor.py`.
4. `executor.py` opens Pitaya and/or EPOS (based on config + CLI flags), iterates points, writes per-point CSVs, and records a `run_manifest.json`.
5. `ui6.py` optionally runs embedded treatment and updates the Analysis tab.

## Configuration Files
`experiment_config.json` includes:
- Sweep ranges and counts.
- Pitaya settings (host, port, n_samples, decimation constraints).
- EPOS plan and safety settings.
- Analysis constants (pressure scale, oscilloscope input resistance).

`measurement_plan.json` includes a list of `grid_points` containing:
- The specific `motor_hz`, `rbox_ohm`, `decimation`, and derived parameters.
- `pitaya_setup` and `pitaya_queries` command lists for the point.

If you add fields in `gen_config.py`, make sure `executor.py` reads them consistently.

## Changing the UI
Common edits live in `ui6.py`:
- Input defaults: look for `self.v_*` assignments near the top of `__init__`.
- Mode behavior: `_on_mode_change()` and `on_run()`.
- Safety checks and RPM limits: `on_run()` and `on_jog()`.
- Log parsing and run-dir discovery: `_maybe_capture_paths()`.

When you add a new UI field that affects configuration:
1. Add a `tk.StringVar` or `tk.BooleanVar` for it.
2. Add a widget to the layout.
3. Plumb into `GenArgs` or executor CLI in `on_run()`.
4. Add validation in `_parse_int/_parse_float` usage.

## Changing Plan Generation
`gen_config.py` owns the experiment grid and Pitaya command planning:
- Add new plan fields by extending `ExperimentConfig` and `build_plan()`.
- Keep `measurement_plan.json` and CSV naming stable unless you intend to update downstream tooling.

## Changing Execution
`executor.py` is responsible for runtime safety and hardware IO:
- EPOS motion logic lives in the main loop and calls into `devices/epos_driver.py`.
- Pitaya acquisition lives in `execute_point()`.
- Safety clamp is enforced by `--rpm-limit`.

If you change the plan schema, update:
- `execute_point()` field access.
- `legacy_csv_name()` if you change naming conventions.

## Data Treatment
Two code paths exist:
- Embedded processing in `ui6.py` (uses the same logic as the standalone tool).
- Standalone tool in `Piezo_data-treatment_Python.py`.

If you modify the CSV format, update both the executor output and the treatment logic.

## Local Development Tips
- Use `--dry-run` in `executor.py` when working without hardware.
- Use small `--points` to keep loops short.
- Run the GUI from repo root so relative paths resolve correctly.
