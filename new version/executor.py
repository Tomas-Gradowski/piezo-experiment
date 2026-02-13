#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from devices.pitaya_scpi import PitayaSCPI
from devices.epos_driver import EposDriver, EposConfig


# -----------------------------
# Parsing helpers
# -----------------------------

def parse_waveform_text(resp: str) -> List[float]:
    s = resp.strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()

    parts = s.split(",") if "," in s else s.split()
    out: List[float] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except ValueError:
            q = "".join(ch for ch in p if ch in "+-0123456789.eEinfNaIN")
            out.append(float(q))
    return out


def make_time_axis(n_samples: int, fs_hz: float, decimation: int) -> List[float]:
    dt = decimation / fs_hz
    return [i * dt for i in range(n_samples)]


# -----------------------------
# IO helpers
# -----------------------------

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_csv_point(out_csv: Path, t: List[float], ch1: List[float], ch2: List[float],
                   meta: Optional[Dict[str, Any]] = None) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        if meta:
            for k, v in meta.items():
                f.write(f"# {k}: {v}\n")
        w = csv.writer(f)
        w.writerow(["time_s", "ch1", "ch2"])
        n = min(len(t), len(ch1), len(ch2))
        for i in range(n):
            w.writerow([t[i], ch1[i], ch2[i]])


def format_digits(d: List[int]) -> str:
    return " ".join(str(x) for x in d)


def resolve_run_dir(p: Path) -> Path:
    if p.is_file():
        p = p.parent

    cfg_name = "experiment_config.json"
    plan_name = "measurement_plan.json"

    if (p / cfg_name).exists() and (p / plan_name).exists():
        return p

    if not p.is_dir():
        raise FileNotFoundError(f"{p} is not a directory and does not contain {cfg_name}/{plan_name}")

    matches: List[Path] = []
    for cfg_path in p.rglob(cfg_name):
        cand = cfg_path.parent
        if (cand / plan_name).exists():
            matches.append(cand)

    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise FileNotFoundError(f"Could not find {cfg_name} + {plan_name} anywhere under {p}.")

    matches.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    print("Warning: multiple run folders found. Using most recent:")
    for m in matches[:10]:
        print(f"  - {m}")
    return matches[0]


# -----------------------------
# Execution core
# -----------------------------

@dataclass
class ExecResult:
    index: int
    ok: bool
    error: Optional[str]
    out_csv: Optional[str]
    n_ch1: int = 0
    n_ch2: int = 0



def execute_point(
    pitaya: PitayaSCPI,
    point: Dict[str, Any],
    fs_hz: float,
    n_samples: int,
    csv_dir: Path,
    no_sleep: bool,
    extra_wait_s: float,
    dry_run: bool,
    meta: Optional[Dict[str, Any]] = None,
) -> ExecResult:

    idx = int(point["index"])
    dec = int(point["decimation"])
    out_csv = csv_dir / f"point_{idx:06d}.csv"

    print(f"\n--- Executing point {idx} ---")
    print(f"Decimation = {dec}")

    if dry_run:
        print("DRY RUN: Skipping hardware communication.")
        print("Would send setup commands:")
        for cmd in point["pitaya_setup"]:
            print("  ", cmd)

        print("Would read channel 1")
        print("Would read channel 2")

        return ExecResult(index=idx, ok=True, error=None,
                          out_csv=str(out_csv),
                          n_ch1=n_samples,
                          n_ch2=n_samples)

    try:
        print("Sending setup commands to Red Pitaya...")
        pitaya.run_commands(point["pitaya_setup"])

        meas_time = (dec / fs_hz) * n_samples + float(extra_wait_s)
        if not no_sleep:
            print(f"Waiting {meas_time:.3f} seconds for acquisition...")
            time.sleep(meas_time)

        print("Reading channel 1 from Red Pitaya...")
        ch1_txt = pitaya.query(point["pitaya_queries"][0])

        print("Reading channel 2 from Red Pitaya...")
        ch2_txt = pitaya.query(point["pitaya_queries"][1])

        ch1 = parse_waveform_text(ch1_txt)
        ch2 = parse_waveform_text(ch2_txt)

        print(f"Channel 1 samples: {len(ch1)}")
        print(f"Channel 2 samples: {len(ch2)}")

        t = make_time_axis(n_samples=n_samples, fs_hz=fs_hz, decimation=dec)

        print(f"Saving CSV to {out_csv}")
        save_csv_point(out_csv, t=t, ch1=ch1, ch2=ch2, meta=meta)

        return ExecResult(index=idx, ok=True, error=None,
                          out_csv=str(out_csv),
                          n_ch1=len(ch1),
                          n_ch2=len(ch2))

    except Exception as e:
        print("ERROR:", e)
        return ExecResult(index=idx, ok=False, error=str(e), out_csv=None)

# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Execute measurement_plan.json on Red Pitaya and export CSV waveforms.")
    ap.add_argument("--run-dir", required=True, help="Folder containing experiment_config.json + measurement_plan.json")
    ap.add_argument("--csv-dirname", default="csv", help="Where to write csv outputs inside run-dir")

    ap.add_argument("--timeout", type=float, default=10.0, help="TCP timeout seconds")
    ap.add_argument("--dry-run", action="store_true", help="Do not connect or run; just show what would happen.")
    ap.add_argument("--no-sleep", action="store_true", help="Skip waiting for acquisition (debug only).")
    ap.add_argument("--extra-wait", type=float, default=0.1, help="Extra wait seconds after acquisition time.")

    ap.add_argument("--start-index", type=int, default=0, help="Start grid point index (inclusive in grid_points list).")
    ap.add_argument("--stop-index", type=int, default=None, help="Stop grid point index (exclusive in grid_points list).")
    ap.add_argument("--points", type=int, default=None, help="Run only first N points from start-index.")
    ap.add_argument("--fail-fast", action="store_true", help="Stop at first failure.")

    ap.add_argument("--prompt-rbox", action="store_true", help="Prompt when r_index changes (manual DR08).")

    ap.add_argument("--use-epos", action="store_true", help="Enable EPOS motor control")
    ap.add_argument("--epos-lib", default="/home/tomas/thesis/epos/EPOS-Linux-Library-En/EPOS_Linux_Library/lib/intel/x86_64/libEposCmd.so.6.8.1.0")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = resolve_run_dir(Path(args.run_dir).expanduser().resolve())
    cfg = load_json(run_dir / "experiment_config.json")
    plan = load_json(run_dir / "measurement_plan.json")

    pitaya_cfg = cfg["pitaya"]
    host = pitaya_cfg["host"]
    port = int(pitaya_cfg["port"])
    fs_hz = float(pitaya_cfg["fs_hz"])
    n_samples = int(pitaya_cfg["n_samples"])

    csv_dir = run_dir / args.csv_dirname
    csv_dir.mkdir(parents=True, exist_ok=True)

    points: List[Dict[str, Any]] = plan["grid_points"]

    start = max(0, int(args.start_index))
    stop = int(args.stop_index) if args.stop_index is not None else len(points)
    stop = min(stop, len(points))
    subset = points[start:stop]
    if args.points is not None:
        subset = subset[: max(0, int(args.points))]

    # Optional DR08 info
    r_entries = plan.get("rbox", {}).get("entries", [])
    current_r_index: Optional[int] = None
    current_r_entry: Optional[Dict[str, Any]] = None

    motor_current = None
    motor_rpm = None
    epos: Optional[EposDriver] = None

    results: List[ExecResult] = []
    t0 = time.perf_counter()

    pitaya = PitayaSCPI(host, port, timeout=args.timeout)

    try:
        if not args.dry_run:
            pitaya.open()

        if args.use_epos and not args.dry_run:
            epos = EposDriver(EposConfig(lib_path=args.epos_lib))
            epos.open()

        for gp in subset:
            r_idx = gp.get("r_index", None)

            # DR08 prompt when resistance changes
            if args.prompt_rbox and r_idx != current_r_index:
                current_r_index = r_idx
                print("\n" + "=" * 70)
                print(f"Set DR08 to R ≈ {gp['rbox_ohm']} Ω (requested {gp['requested_r_ohm']} Ω)")
                print("Digits [1M,100k,10k,1k,100,10,1,0.1]:")
                print(format_digits(gp["rbox_digits"]))
                print("Set the DR08 now, then press Enter to continue...")
                input()
            if epos is not None:
                motor_current = epos.get_current_mA()

            # metadata per point (goes into commented header)
            meta = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "index": gp.get("index"),
                "r_index": gp.get("r_index"),
                "f_index": gp.get("f_index"),
                "freq_hz": gp.get("freq_hz"),
                "decimation": gp.get("decimation"),
                "motor_current_mA": motor_current,
            }
            if current_r_entry is not None:
                meta.update({
                    "requested_r_ohm": current_r_entry.get("requested_ohm"),
                    "rbox_ohm": current_r_entry.get("rbox_ohm"),
                    "rbox_digits": current_r_entry.get("digits"),
                })

            res = execute_point(
                pitaya=pitaya,
                point=gp,
                fs_hz=fs_hz,
                n_samples=n_samples,
                csv_dir=csv_dir,
                no_sleep=args.no_sleep,
                extra_wait_s=args.extra_wait,
                dry_run=args.dry_run,
                meta=meta,
            )
            results.append(res)

            if res.ok:
                print(f"Saved {res.out_csv} (n={res.n_ch1})")
            else:
                print(f"FAIL point {res.index}: {res.error}")
                if args.fail_fast:
                    break

    finally:
        if epos is not None:
            epos.close()
        if not args.dry_run:
            pitaya.close()

    dt = time.perf_counter() - t0

    manifest = {
        "run_dir": str(run_dir),
        "pitaya": {"host": host, "port": port, "timeout": args.timeout},
        "n_samples": n_samples,
        "fs_hz": fs_hz,
        "executed_points": len(results),
        "ok": sum(1 for r in results if r.ok),
        "fail": sum(1 for r in results if not r.ok),
        "elapsed_s": dt,
        "results": [r.__dict__ for r in results],
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote manifest: {run_dir / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
