# Piezo Experiment Pipeline

This repo splits the workflow into two independent steps:

1. **testing/**: acquisition scripts (hardware + Red Pitaya interaction)
2. **plotting/**: analysis scripts (offline, reads local `runs/` data)

## Quick start
- Put experiment parameters in `config/example.yaml`
- Run acquisition:
  - `python testing/testing.py --config config/example.yaml`
- Analyze a run:
  - `python plotting/plotting.py --run runs/<RUN_FOLDER>`
