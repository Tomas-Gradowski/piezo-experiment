#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from devices.pitaya_scpi import scpi as PitayaSCPI
from devices.epos_driver import EposDriver, EposConfig, Direction 


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


def save_csv_point(
    out_csv: Path,
    t: List[float],
    pressure: List[float],
    voltage: List[float],
    power: List[float],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Match legacy C# CSV layout exactly.
        w.writerow(["time", "pressure", "voltage", "power"])
        n = min(len(t), len(pressure), len(voltage), len(power))
        for i in range(n):
            w.writerow([t[i], pressure[i], voltage[i], power[i]])


def format_legacy_num(v: float) -> str:
    return f"{float(v):g}"


def legacy_csv_name(point: Dict[str, Any], rep_index: int) -> str:
    freq = float(point.get("motor_hz", point.get("signal_hz", 0.0)))
    r_ohm = float(point.get("rbox_ohm", point.get("requested_r_ohm", 0.0)))
    return f"SingleMeasurement_{format_legacy_num(freq)}Hz_{format_legacy_num(r_ohm)}Ohms_{rep_index}.csv"


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
    pitaya: Optional[PitayaSCPI],
    point: Dict[str, Any],
    fs_hz: float,
    n_samples: int,
    csv_dir: Path,
    no_sleep: bool,
    extra_wait_s: float,
    dry_run: bool,
    meta: Optional[Dict[str, Any]] = None,
    rep_index: int = 0,
    use_pitaya: bool = True,
    pressure_scale: float = 0.5,
    osc_r_ohm: float = 1e6,
) -> ExecResult:

    idx = int(point["index"])
    dec = int(point["decimation"])
    out_csv = csv_dir / legacy_csv_name(point=point, rep_index=rep_index)

    print(f"\n--- Executing point {idx} ---")
    print(f"Decimation = {dec}")
    if not use_pitaya:
        print("Pitaya disabled: skipping acquisition")
        return ExecResult(index=idx, ok=True, error=None, out_csv=None, n_ch1=0, n_ch2=0)
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
        if pitaya is None:
            raise RuntimeError("Pitaya client not initialized")
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

        # Legacy C# mapping:
        # pressure = 0.5 * CH1
        # voltage  = CH2
        # power    = voltage^2 / totalR
        freq_hz = float(point.get("motor_hz", point.get("signal_hz", 0.0)))
        load_r = float(point.get("rbox_ohm", point.get("requested_r_ohm", 0.0)))
        z_osc = float(osc_r_ohm)
        if freq_hz > 0:
            z_osc = math.sqrt((float(osc_r_ohm) ** 2) + (((1.0 / freq_hz) * 2.0 * math.pi * 10e-9) ** 2))
        if load_r > 0 and z_osc > 0:
            total_r = 1.0 / (1.0 / load_r + 1.0 / z_osc)
        else:
            total_r = max(z_osc, 1e-12)

        pressure = [float(pressure_scale) * p for p in ch1]
        voltage = [float(v) for v in ch2]
        power = [(v * v) / total_r for v in voltage]

        t = make_time_axis(n_samples=len(ch1), fs_hz=fs_hz, decimation=dec)
        print(f"Saving CSV to {out_csv}")
        save_csv_point(out_csv, t=t, pressure=pressure, voltage=voltage, power=power)

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
    ap.add_argument("--pitaya-host", default=None, help="Override Pitaya host from config (e.g. 192.168.x.x)")
    ap.add_argument("--pitaya-port", type=int, default=None, help="Override Pitaya TCP port from config")
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
    
    ap.add_argument("--no-pitaya", action="store_true", help="Force-disable Pitaya acquisition (EPOS-only run)")

    # Manual EPOS jog test (does not run the plan loop)
    ap.add_argument("--epos-jog", action="store_true", help="Run a manual EPOS jog test then exit")
    ap.add_argument("--epos-jog-dir", choices=["cw", "ccw"], default="cw", help="Direction for manual jog")
    ap.add_argument("--epos-jog-rpm", type=int, default=30, help="Output rpm for manual jog")
    ap.add_argument("--epos-jog-seconds", type=float, default=2.0, help="How long to jog")
    ap.add_argument("--rpm-limit", type=int, default=5000, help="Hard safety cap ")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = resolve_run_dir(Path(args.run_dir).expanduser().resolve())
    cfg = load_json(run_dir / "experiment_config.json")
    plan = load_json(run_dir / "measurement_plan.json")

    pitaya_cfg = cfg.get("pitaya", {})
    epos_cfg = cfg.get("epos", {})

    # Enable flags (cfg + CLI)
    use_pitaya = bool(pitaya_cfg.get("enabled", True)) and (not args.no_pitaya)
    use_epos_cfg = bool(epos_cfg.get("enabled", False))
    use_epos = bool(args.use_epos) and use_epos_cfg

    # Pitaya parameters (safe defaults)
    host = args.pitaya_host or pitaya_cfg.get("host", "rp-f06549.local")
    port = int(args.pitaya_port) if args.pitaya_port is not None else int(pitaya_cfg.get("port", 5000))
    fs_hz = float(pitaya_cfg.get("fs_hz", 125e6))
    n_samples = int(pitaya_cfg.get("n_samples", 16384))
    pressure_scale = float(cfg.get("pressure_scale", 0.5))
    osc_r_ohm = float(cfg.get("osc_r_ohm", 1e6))

    csv_dir = run_dir / args.csv_dirname
    csv_dir.mkdir(parents=True, exist_ok=True)

    points: List[Dict[str, Any]] = plan["grid_points"]

    start = max(0, int(args.start_index))
    stop = int(args.stop_index) if args.stop_index is not None else len(points)
    stop = min(stop, len(points))
    subset = points[start:stop]
    if args.points is not None:
        subset = subset[: max(0, int(args.points))]

    # DR08 prompt tracking
    current_r_index: Optional[int] = None

    # EPOS run parameters (from cfg["epos"])
    repetitions = int(cfg.get("repetitions", 1))
    periods = float(cfg.get("periods", 10.0))

    epos_direction = str(epos_cfg.get("direction", "cw"))
    epos_alternate = bool(epos_cfg.get("alternate", False))
    epos_rest_s = float(epos_cfg.get("rest_s", 0.0))
    epos_rpm_override = int(epos_cfg.get("rpm_override", 0))

    results: List[ExecResult] = []
    t0 = time.perf_counter()

    pitaya: Optional[PitayaSCPI] = None
    epos: Optional[EposDriver] = None
    stop_requested = False

    def _request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"[Signal] received {signum}; stopping...")
        if epos is not None:
            try:
                epos.jog_stop()
            except Exception:
                pass

    prev_sigint = signal.signal(signal.SIGINT, _request_stop)
    prev_sigterm = signal.signal(signal.SIGTERM, _request_stop)

    try:
        # --- Open Pitaya
        if use_pitaya and (not args.dry_run):
            pitaya = PitayaSCPI(host, port=port, timeout=args.timeout)
            pitaya.open()
        else:
            print("Pitaya is disabled or dry-run: not opening Red Pitaya.")

        # --- Open EPOS (prefer config JSON for lib_path)
        if use_epos and (not args.dry_run):
            lib_path = epos_cfg.get("lib_path", args.epos_lib)
            epos = EposDriver(EposConfig(
                lib_path=lib_path,
                node_id=int(epos_cfg.get("node_id", 1)),
                gear_reduction=int(epos_cfg.get("gear_reduction", 18)),
                one_turn=int(epos_cfg.get("one_turn", 73728)),
                max_rpm=int(epos_cfg.get("max_rpm", 5000)),
                accel=int(epos_cfg.get("accel", 250000)),
                decel=int(epos_cfg.get("decel", 250000)),
            ))
            print("Opening EPOS...")
            epos.open()
        else:
            if args.use_epos:
                print("EPOS requested but disabled in config or dry-run: not opening EPOS.")

        # --- Manual EPOS jog mode (test motor without experiment)
        if args.epos_jog:
            jog_rpm = max(0, min(int(args.epos_jog_rpm), int(args.rpm_limit)))
            if not args.use_epos:
                print("ERROR: --epos-jog requires --use-epos")
                return

            if args.dry_run:
                print(f"[EPOS dry-run jog] dir={args.epos_jog_dir} rpm_out={jog_rpm} seconds={args.epos_jog_seconds}")
                return

            if epos is None:
                raise RuntimeError("EPOS not opened (is it enabled in experiment_config.json and --use-epos passed?)")

            d = Direction(args.epos_jog_dir)
            print(f"[EPOS jog] START dir={d.value} rpm_out={jog_rpm} seconds={args.epos_jog_seconds}")
            motor_rpm = epos.jog_start(rpm_output=jog_rpm, direction=d)
            eff_out_rpm = abs(motor_rpm) / max(1, int(epos.cfg.gear_reduction))
            print(f"[EPOS jog] effective_motor_rpm={motor_rpm} effective_output_rpm={eff_out_rpm:.2f}")
            t_end = time.perf_counter() + max(0.0, float(args.epos_jog_seconds))
            while time.perf_counter() < t_end:
                if stop_requested:
                    break
                remaining = t_end - time.perf_counter()
                time.sleep(min(0.02, max(0.0, remaining)))
            epos.jog_stop()
            print("[EPOS jog] STOP")
            return

        # --- Main run loop
        for gp in subset:
            if stop_requested:
                print("[STOP] interrupted before point execution")
                break
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

            for rep in range(repetitions):
                if stop_requested:
                    print("[STOP] interrupted before repetition")
                    break
                # --- EPOS motion (before acquisition)
                motor_current = None
                if use_epos:
                    d_str = epos_direction
                    if epos_alternate and (rep % 2 == 1):
                        d_str = "ccw" if epos_direction == "cw" else "cw"

                    motor_hz = float(gp["motor_hz"])
                    rpm_out_plan = int(gp["rpm_out"])

                    dur_s = periods / motor_hz
                    rpm_out = epos_rpm_override if epos_rpm_override > 0 else rpm_out_plan
                    rpm_out = max(0,min(rpm_out, int(args.rpm_limit)))

                    if args.dry_run:
                        print(f"[EPOS dry-run] point={gp['index']} rep={rep} dir={d_str} rpm_out={rpm_out} duration_s={dur_s:.3f}")
                    else:
                        assert epos is not None
                        d = Direction(d_str)
                        print(f"[EPOS] point={gp['index']} rep={rep} dir={d.value} rpm_out={rpm_out} duration_s={dur_s:.3f}")
                        motor_rpm = epos.move_for_time(
                            duration_s=dur_s,
                            rpm_output=rpm_out,
                            direction=d,
                            should_stop=lambda: stop_requested,
                        )
                        eff_out_rpm = abs(motor_rpm) / max(1, int(epos.cfg.gear_reduction))
                        print(f"[EPOS] effective_motor_rpm={motor_rpm} effective_output_rpm={eff_out_rpm:.2f}")
                        motor_current = epos.get_current_mA()
                        vel = epos.get_velocity_rpm()
                        print(f"[EPOS] current_mA={motor_current} vel_rpm={vel}")
                        if epos_rest_s > 0:
                            time.sleep(epos_rest_s)

                # --- Metadata per repetition
                meta = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "index": gp.get("index"),
                    "rep": rep,
                    "r_index": gp.get("r_index"),
                    "f_index": gp.get("f_index"),
                    "motor_hz": gp.get("motor_hz"),
                    "signal_hz": gp.get("signal_hz"),
                    "decimation": gp.get("decimation"),
                    "motor_current_mA": motor_current,
                }

                # --- Pitaya acquisition per repetition (or skip)
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
                    rep_index=rep,
                    use_pitaya=use_pitaya,
                    pressure_scale=pressure_scale,
                    osc_r_ohm=osc_r_ohm,
                )
                results.append(res)

                if res.ok:
                    if res.out_csv:
                        print(f"Saved {res.out_csv} (n={res.n_ch1})")
                    else:
                        print("OK (no CSV written)")
                else:
                    print(f"FAIL point {res.index}: {res.error}")
                    if args.fail_fast:
                        break

            if args.fail_fast and results and (not results[-1].ok):
                break

    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)
        if epos is not None:
            try:
                epos.jog_stop()
            except Exception:
                pass
            epos.close()

        if (not args.dry_run) and use_pitaya and pitaya is not None:
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
