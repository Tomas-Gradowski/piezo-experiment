# Experiment Configuration Generator

This repository contains tools to define, execute, and analyze PVDF / piezoelectric experiments.
This script replaces the original C# WinForms UI previously used to configure experiments.

The goal is to provide **reproducible, UI-independent experiment definitions** that can be
executed and analyzed consistently.

---

## Configuration Generator (CLI)

The configuration generator is a command-line Python script that produces:

- a full experiment configuration file
- a per-measurement execution plan
- a test Red Pitaya setup script

It **does not communicate with hardware**.  
It only defines *what* an experiment is, not *how* it is executed.

**Script location:**

```

config/scripts/generate_experiment_config.py

```

---

## What the script generates

For each run, the script creates a directory:

```

<out-dir>/<timestamp>/<sample-name>/
├── experiment_config.json
├── measurement_plan.json
└── pitaya_setup_first_point.txt

````

### `experiment_config.json`
Human-readable configuration containing:
- frequency and resistance sweep definitions
- acquisition parameters
- calibration constants
- metadata (sample name, creation time, output directory)

This file is the **single source of truth** for experiment parameters.

### `measurement_plan.json`
Machine-readable execution plan containing:
- one entry per (frequency, resistance) grid point
- computed decimation per point
- Red Pitaya SCPI command lists
- acquisition queries

This file is intended to be consumed by acquisition scripts.

### `pitaya_setup_first_point.txt`
Convenience file containing the Red Pitaya commands for the first grid point.
Useful for quick manual testing via telnet or netcat.

---

## How to run the script

### Basic example

```bash
python3 config/scripts/generate_experiment_config.py \
  --sample-name "PVDF_test_01" \
  --out-dir "~/" \
  --min-freq 5 --max-freq 40 --freq-points 8 --freq-scale linear \
  --min-r 1000 --max-r 1000000 --r-points 10 --r-scale log \
  --periods 10 \
  --repetitions 3
````

The output directory supports `~` expansion.

---

## Required arguments

### Experiment metadata

* `--sample-name`
  Name of the sample / experiment.

* `--out-dir`
  Base directory where the run folder will be created.

### Frequency sweep

* `--min-freq` (Hz)
* `--max-freq` (Hz)
* `--freq-points`
* `--freq-scale` : `linear` or `log`

### Resistance sweep

* `--min-r` (Ohm)
* `--max-r` (Ohm)
* `--r-points`
* `--r-scale` : `linear` or `log`

---

## Common optional arguments

* `--periods` (default: `10`)
  Number of signal periods captured per acquisition.
  Used to compute the Red Pitaya decimation.

* `--repetitions` (default: `1`)
  Number of repeated acquisitions per (R, f) grid point.

---

## Red Pitaya / acquisition parameters (optional)

Defaults reproduce the original C# behavior.

* `--pitaya-host` (default: `rp-f06549.local`)
* `--pitaya-port` (default: `5000`)
* `--fs-hz` (default: `125e6`)
* `--n-samples` (default: `16384`)
* `--trig-delay` (default: `8192`)
* `--gain-ch1` (default: `HV`)
* `--gain-ch2` (default: `HV`)
* `--units` (default: `VOLTS`)

---

## Calibration / model parameters (optional)

* `--pressure-scale` (default: `0.5`)
  Pressure calibration factor applied during analysis.

* `--osc-r-ohm` (default: `1e6`)
  Equivalent input resistance used to compute the effective load resistance:

  ```
  R_total = 1 / (1/R + 1/osc_r)
  ```

* `--max-decimation` (default: `65536`)

---

## Design principles

* No UI logic
* No hardware access
* Fully reproducible experiments
* Clear separation between:

  * configuration
  * execution
  * analysis

This script is intended to be the **single authoritative definition** of an experiment.

---

## Typical workflow

1. Generate configuration and execution plan using this script
2. Run acquisition using `measurement_plan.json`
3. Analyze raw shots using `experiment_config.json` + analysis tools

---


