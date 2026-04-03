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

HARD_MAX_OUTPUT_RPM = 2000


# -----------------------------
# Parsing helpers
# -----------------------------

def parse_waveform_text(resp: str) -> List[float]:
    s = (resp or "").strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    if not s:
        return []

    out: List[float] = []
    for part in s.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            out.append(float(token))
        except ValueError:
            raise ValueError(f"Malformed SCPI waveform token: {token!r}")
    return out


def make_time_axis(n_samples: int, fs_hz: float, decimation: int) -> List[float]:
    dt = decimation / fs_hz
    return [i * dt for i in range(n_samples)]

def normalize_pitaya_queries(queries: Optional[List[str]]) -> List[str]:
    # Legacy expects CH1=pressure and CH2=voltage.
    if not queries or len(queries) < 2:
        return ["ACQ:SOUR1:DATA?", "ACQ:SOUR2:DATA?"]

    q1 = (queries[0] or "").strip()
    q2 = (queries[1] or "").strip()

    # Fix known typo seen in older generated plans.
    if "commandUR1" in q1:
        q1 = "ACQ:SOUR1:DATA?"
    if "commandUR2" in q2:
        q2 = "ACQ:SOUR2:DATA?"

    # Ensure proper channels are queried.
    if "SOUR1" not in q1.upper():
        q1 = "ACQ:SOUR1:DATA?"
    if "SOUR2" not in q2.upper():
        q2 = "ACQ:SOUR2:DATA?"

    return [q1, q2]


def normalize_pitaya_setup(commands: Optional[List[str]]) -> List[str]:
    if not commands:
        return [
            "ACQ:RST",
            "ACQ:DATA:UNITS VOLTS",
            "ACQ:SOUR1:GAIN HV",
            "ACQ:SOUR2:GAIN HV",
            "ACQ:START",
            "ACQ:TRIG NOW",
        ]

    cmds = [c.strip() for c in commands if c and c.strip()]
    upper = [c.upper() for c in cmds]

    def _has(prefix: str) -> bool:
        return any(c.startswith(prefix) for c in upper)

    # Keep plan-provided DEC and TRIG settings but ensure core config exists.
    if not _has("ACQ:DATA:UNITS"):
        cmds.insert(1, "ACQ:DATA:UNITS VOLTS")
    if not _has("ACQ:SOUR1:GAIN"):
        cmds.insert(2, "ACQ:SOUR1:GAIN HV")
    if not _has("ACQ:SOUR2:GAIN"):
        cmds.insert(3, "ACQ:SOUR2:GAIN HV")
    if not _has("ACQ:START"):
        cmds.append("ACQ:START")
    if not any(c.startswith("ACQ:TRIG") for c in upper):
        cmds.append("ACQ:TRIG NOW")

    return cmds


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


def append_summary_row(
    summary_csv: Path,
    *,
    index: int,
    rep_index: int,
    freq_hz: float,
    r_ohm: float,
    pressure_amp: float,
    voltage_amp: float,
) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_csv.exists()
    with summary_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "index",
                "rep",
                "freq_hz",
                "r_ohm",
                "pressure_amp",
                "voltage_mean_abs",
            ])
        w.writerow([index, rep_index, freq_hz, r_ohm, pressure_amp, voltage_amp])


def mean_abs(values: List[float]) -> float:
    return (sum(abs(v) for v in values) / len(values)) if values else 0.0


def _effective_scales(cfg: Dict[str, Any]) -> tuple[float, float]:
    v_scale = float(cfg.get("voltage_scale", 1.0))
    probe_factor = 10.0 if bool(cfg.get("probe_10x", False)) else 1.0
    return v_scale, probe_factor


def _pressure_model_coeffs(cfg: Dict[str, Any]) -> Optional[tuple[float, float]]:
    # Returns (a, b) for pressure_bar = a * V + b
    if not all(k in cfg for k in ("pressure_v_min", "pressure_v_max", "pressure_p_min", "pressure_p_max")):
        return None
    v_min = float(cfg.get("pressure_v_min", 0.0))
    v_max = float(cfg.get("pressure_v_max", 0.0))
    p_min = float(cfg.get("pressure_p_min", 0.0))
    p_max = float(cfg.get("pressure_p_max", 0.0))
    if v_max == v_min:
        return None
    a = (p_max - p_min) / (v_max - v_min)
    b = p_min - a * v_min
    return a, b




def format_legacy_num(v: float) -> str:
    return f"{float(v):g}"


def legacy_csv_name(point: Dict[str, Any], rep_index: int, freq_hz_override: Optional[float] = None) -> str:
    if freq_hz_override is None:
        freq = float(point.get("output_hz", point.get("motor_hz", point.get("signal_hz", 0.0))))
    else:
        freq = float(freq_hz_override)
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
    pressure_cal_a: float = 0.5,
    pressure_cal_b: float = 0.0,
    voltage_cal_a: float = 1.0,
    voltage_cal_b: float = 0.0,
    probe_factor: float = 1.0,
    osc_r_ohm: float = 1e6,
    freq_hz_override: Optional[float] = None,
) -> ExecResult:

    idx = int(point["index"])
    dec = int(point["decimation"])
    out_csv = csv_dir / legacy_csv_name(point=point, rep_index=rep_index, freq_hz_override=freq_hz_override)
    setup_cmds = point.get("pitaya_setup")
    queries = point.get("pitaya_queries")

    print(f"\n--- Executing point {idx} ---")
    print(f"Decimation = {dec}")
    if not use_pitaya:
        print("Pitaya disabled: skipping acquisition")
        return ExecResult(index=idx, ok=True, error=None, out_csv=None, n_ch1=0, n_ch2=0)
    if dry_run:
        print("DRY RUN: Skipping hardware communication.")
        print("Would send setup commands:")
        for cmd in setup_cmds:
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
        pitaya.run_commands(setup_cmds)

        meas_time = (dec / fs_hz) * n_samples + float(extra_wait_s)
        wait_for_pitaya_trigger(pitaya, timeout_s=min(2.0, max(0.5, meas_time)))
        if not no_sleep:
            print(f"Waiting {meas_time:.3f} seconds for acquisition...")
            time.sleep(meas_time)

        print("Reading channel 1 from Red Pitaya...")
        ch1 = _query_waveform_with_retry(pitaya, queries[0], "ch1")

        print("Reading channel 2 from Red Pitaya...")
        ch2 = _query_waveform_with_retry(pitaya, queries[1], "ch2")

        if not ch1 or not ch2:
            # Re-arm acquisition and retry once for both channels.
            print("[Pitaya] empty channel data; re-arming acquisition and retrying...")
            pitaya.run_commands(["ACQ:STOP", "ACQ:RST"])
            pitaya.run_commands(setup_cmds)
            wait_for_pitaya_trigger(pitaya, timeout_s=min(2.0, max(0.5, meas_time)))
            if not no_sleep:
                time.sleep(max(0.05, min(0.5, meas_time)))

            print("Reading channel 1 from Red Pitaya (after re-arm)...")
            ch1 = _query_waveform_with_retry(pitaya, queries[0], "ch1")
            print("Reading channel 2 from Red Pitaya (after re-arm)...")
            ch2 = _query_waveform_with_retry(pitaya, queries[1], "ch2")

        if not ch1 or not ch2:
            raise RuntimeError(
                "No numeric waveform samples parsed "
                f"(ch1={len(ch1)}, ch2={len(ch2)}). "
                "Check SCPI response integrity and trigger/acquisition timing."
            )

        print(f"Channel 1 samples: {len(ch1)}")
        print(f"Channel 2 samples: {len(ch2)}")
        if ch1:
            print(f"Channel 1 min/max: {min(ch1):.6g} / {max(ch1):.6g}")
        if ch2:
            print(f"Channel 2 min/max: {min(ch2):.6g} / {max(ch2):.6g}")

        # Legacy C# mapping:
        # pressure = 0.5 * CH1
        # voltage  = CH2
        # power    = voltage^2 / totalR
        freq_hz = float(freq_hz_override) if freq_hz_override is not None else float(
            point.get("output_hz", point.get("motor_hz", point.get("signal_hz", 0.0)))
        )
        load_r = float(point.get("rbox_ohm", point.get("requested_r_ohm", 0.0)))
        total_r = float(point.get("total_r_ohm", 0.0))
        if total_r <= 0.0:
            z_osc = float(osc_r_ohm)
            if load_r > 0 and z_osc > 0:
                total_r = 1.0 / (1.0 / load_r + 1.0 / z_osc)
            else:
                total_r = max(z_osc, 1e-12)

        pressure = [(float(pressure_cal_a) * (p * probe_factor)) + float(pressure_cal_b) for p in ch1]
        voltage = [(float(voltage_cal_a) * (v * probe_factor)) + float(voltage_cal_b) for v in ch2]
        power = [(v * v) / total_r for v in voltage]

        t = make_time_axis(n_samples=len(ch1), fs_hz=fs_hz, decimation=dec)
        print(f"Saving CSV to {out_csv}")
        save_csv_point(out_csv, t=t, pressure=pressure, voltage=voltage, power=power)
        if pressure and voltage:
            p_min, p_max = min(pressure), max(pressure)
            p_amp = (p_max - p_min)
            v_amp = mean_abs(voltage)
            summary_csv = csv_dir / "summary.csv"
            append_summary_row(
                summary_csv,
                index=idx,
                rep_index=rep_index,
                freq_hz=freq_hz,
                r_ohm=load_r,
                pressure_amp=p_amp,
                voltage_amp=v_amp,
            )

        return ExecResult(index=idx, ok=True, error=None,
                          out_csv=str(out_csv),
                          n_ch1=len(ch1),
                          n_ch2=len(ch2))

    except Exception as e:
        print("ERROR:", e)
        return ExecResult(index=idx, ok=False, error=str(e), out_csv=None)


def _is_pitaya_transient_error(msg: str) -> bool:
    m = msg.lower()
    return (
        "scpi socket not connected" in m
        or "failed to connect to red pitaya" in m
        or "connection reset" in m
        or "broken pipe" in m
        or "timed out" in m
        or "timeout" in m
    )


def wait_for_pitaya_trigger(pitaya: PitayaSCPI, timeout_s: float = 1.5, poll_s: float = 0.05) -> bool:
    deadline = time.perf_counter() + max(0.05, float(timeout_s))
    last: Optional[str] = None
    while time.perf_counter() < deadline:
        try:
            resp = pitaya.query("ACQ:TRIG:STAT?")
        except Exception as e:
            print(f"[Pitaya] trigger status query failed: {e}")
            return False
        last = (resp or "").strip()
        if "TD" in last:
            return True
        time.sleep(max(0.01, float(poll_s)))
    if last is not None:
        print(f"[Pitaya] trigger not ready after {timeout_s:.2f}s (last status={last!r}); continuing.")
    return False

def _query_waveform_with_retry(
    pitaya: PitayaSCPI,
    query: str,
    label: str,
    retry_sleep_s: float = 0.15,
    retries: int = 1,
) -> List[float]:
    def _sanitize_waveform_text(txt: str) -> str:
        if not txt:
            return ""
        s = txt.strip()
        # Some Pitaya responses can return trigger status instead of waveform data.
        if s in {"TD", "WAIT", "NONE"}:
            return ""
        # If a braced payload exists, prefer that slice.
        if "{" in s and "}" in s:
            start = s.find("{")
            end = s.find("}", start)
            if start >= 0 and end > start:
                return s[start:end + 1]
        return s

    attempts = 1 + max(0, int(retries))
    for i in range(attempts):
        txt = pitaya.query(query)
        cleaned = _sanitize_waveform_text(txt)
        if not cleaned:
            if i + 1 < attempts:
                time.sleep(max(0.01, float(retry_sleep_s)))
                continue
            return []
        try:
            data = parse_waveform_text(cleaned)
        except ValueError:
            # Malformed token (e.g., "TD") or partial payload: retry.
            data = []
        if data:
            return data
        if i + 1 < attempts:
            time.sleep(max(0.01, float(retry_sleep_s)))

    return []

# -----------------------------
# CLI
# -----------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
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
    ap.add_argument("--pitaya-diag", action="store_true", help="Query basic Pitaya status and exit")

    # Manual EPOS jog test (does not run the plan loop)
    ap.add_argument("--epos-jog", action="store_true", help="Run a manual EPOS jog test then exit")
    ap.add_argument("--epos-jog-dir", choices=["cw", "ccw"], default="cw", help="Direction for manual jog")
    ap.add_argument("--epos-jog-rpm", type=int, default=30, help="Output rpm for manual jog")
    ap.add_argument("--epos-jog-seconds", type=float, default=2.0, help="How long to jog")
    ap.add_argument("--rpm-limit", type=int, default=5000, help="Hard safety cap ")
    return ap.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

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
    host = args.pitaya_host or pitaya_cfg.get("host", "169.254.93.42")
    port = int(args.pitaya_port) if args.pitaya_port is not None else int(pitaya_cfg.get("port", 5000))
    fs_hz = float(pitaya_cfg.get("fs_hz", 125e6))
    n_samples = int(pitaya_cfg.get("n_samples", 16384))
    voltage_scale = float(cfg.get("voltage_scale", 1.0))
    probe_10x = bool(cfg.get("probe_10x", False))
    probe_factor = 10.0 if probe_10x else 1.0
    osc_r_ohm = float(cfg.get("osc_r_ohm", 1e6))

    pressure_cal_a = 0.5
    pressure_cal_b = 0.0
    voltage_cal_a = voltage_scale
    voltage_cal_b = 0.0

    # Use explicit sensor model from config (e.g., 0-5V => 0-2.5 bar).
    model = _pressure_model_coeffs(cfg)
    if model is not None:
        pressure_cal_a, pressure_cal_b = model
    else:
        # Fallback to Gems 3500 defaults if missing.
        pressure_cal_a, pressure_cal_b = _pressure_model_coeffs({
            "pressure_v_min": 0.0,
            "pressure_v_max": 5.0,
            "pressure_p_min": 0.0,
            "pressure_p_max": 2.5,
        }) or (0.5, 0.0)

    print(
        "[Calibration] using pressure model "
        f"(v_min={cfg.get('pressure_v_min', 0.0)} v_max={cfg.get('pressure_v_max', 5.0)} "
        f"p_min={cfg.get('pressure_p_min', 0.0)} p_max={cfg.get('pressure_p_max', 2.5)}) => "
        f"pressure_fit=({pressure_cal_a:.6g}*V + {pressure_cal_b:.6g})"
    )

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
    gear_reduction = max(1, int(epos_cfg.get("gear_reduction", 1)))
    if "max_output_rpm" in epos_cfg:
        max_out_rpm = max(1, int(epos_cfg.get("max_output_rpm", 300)))
    else:
        max_motor_rpm = max(1, int(epos_cfg.get("max_rpm", 5000)))
        max_out_rpm = max(1, max_motor_rpm // max(1, gear_reduction))
    max_motor_rpm = max_out_rpm * gear_reduction
    if max_out_rpm > HARD_MAX_OUTPUT_RPM:
        max_out_rpm = HARD_MAX_OUTPUT_RPM
    eff_out_cap = min(int(args.rpm_limit), max_out_rpm, HARD_MAX_OUTPUT_RPM)
    eff_motor_cap = eff_out_cap * gear_reduction

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
            idn = pitaya.query("*IDN?").strip()
            if not idn:
                raise RuntimeError(f"Pitaya handshake failed on {host}:{port}: empty *IDN? response")
            print(f"[Pitaya] connected {host}:{port} *IDN? => {idn}")
        else:
            print("Pitaya is disabled or dry-run: not opening Red Pitaya.")

        if args.pitaya_diag:
            if pitaya is None:
                raise RuntimeError("Pitaya not opened (is it enabled and not dry-run?)")
            print("[Pitaya diag] ACQ:DEC? =>", pitaya.query("ACQ:DEC?").strip())
            print("[Pitaya diag] ACQ:TRIG:STAT? =>", pitaya.query("ACQ:TRIG:STAT?").strip())
            print("[Pitaya diag] ACQ:TRIG:DLY? =>", pitaya.query("ACQ:TRIG:DLY?").strip())
            print("[Pitaya diag] ACQ:SOUR1:GAIN? =>", pitaya.query("ACQ:SOUR1:GAIN?").strip())
            print("[Pitaya diag] ACQ:SOUR2:GAIN? =>", pitaya.query("ACQ:SOUR2:GAIN?").strip())
            return

        # --- Open EPOS (prefer config JSON for lib_path)
        if use_epos and (not args.dry_run):
            lib_path = epos_cfg.get("lib_path", args.epos_lib)
            epos = EposDriver(EposConfig(
                lib_path=lib_path,
                node_id=int(epos_cfg.get("node_id", 1)),
                gear_reduction=gear_reduction,
                one_turn=int(epos_cfg.get("one_turn", 73728)),
                max_output_rpm=max_out_rpm,
                accel=int(epos_cfg.get("accel", 250000)),
                decel=int(epos_cfg.get("decel", 250000)),
            ))
            print("Opening EPOS...")
            epos.open()
        else:
            if args.use_epos:
                print("EPOS requested but disabled in config or dry-run: not opening EPOS.")

        if use_epos:
            print(
                "[EPOS limits] "
                f"gear_reduction={gear_reduction} "
                f"motor_max_rpm={max_motor_rpm} "
                f"output_max_rpm={max_out_rpm} "
                f"rpm_limit={int(args.rpm_limit)} "
                f"effective_output_cap={eff_out_cap} "
                f"effective_motor_cap={eff_motor_cap}"
            )

        # --- Manual EPOS jog mode (test motor without experiment)
        if args.epos_jog:
            jog_rpm = max(0, min(int(args.epos_jog_rpm), int(eff_out_cap)))
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
                freq_hz_effective: Optional[float] = None
                output_hz = float(gp.get("output_hz", gp.get("motor_hz", 0.0)))
                rpm_out = 0
                d = None
                motor_rpm = 0
                if use_epos:
                    d_str = epos_direction
                    if epos_alternate and (rep % 2 == 1):
                        d_str = "ccw" if epos_direction == "cw" else "cw"

                    rpm_out_plan = int(gp.get("output_rpm", gp.get("rpm_out", 0)))

                    rpm_out = epos_rpm_override if epos_rpm_override > 0 else rpm_out_plan
                    rpm_out = max(0, min(rpm_out, int(eff_out_cap)))
                    # Tie experiment frequency to effective output rpm (same convention as jog).
                    # 1 Hz => 60 rpm output.
                    freq_hz_effective = (float(rpm_out) / 60.0) if rpm_out > 0 else 0.0

                    if args.dry_run:
                        print(
                            f"[EPOS dry-run] point={gp['index']} rep={rep} dir={d_str} "
                            f"rpm_out={rpm_out} freq_hz={freq_hz_effective:.6g}"
                        )
                    else:
                        assert epos is not None
                        d = Direction(d_str)
                        print(
                            f"[EPOS] point={gp['index']} rep={rep} dir={d.value} "
                            f"rpm_out={rpm_out} freq_hz={freq_hz_effective:.6g}"
                        )
                        motor_rpm = epos.jog_start(rpm_output=rpm_out, direction=d)
                        eff_out_rpm = abs(motor_rpm) / max(1, int(epos.cfg.gear_reduction))
                        print(f"[EPOS] effective_motor_rpm={motor_rpm} effective_output_rpm={eff_out_rpm:.2f}")
                        # Match legacy behavior: wait for target speed before acquisition (with timeout).
                        target = max(0, int(abs(motor_rpm)))
                        timeout_s = 3.0
                        t0 = time.perf_counter()
                        vel = epos.get_velocity_rpm()
                        if vel is None:
                            time.sleep(0.5)
                        else:
                            while (time.perf_counter() - t0) < timeout_s:
                                vel = epos.get_velocity_rpm()
                                if vel is None:
                                    break
                                if abs(vel) >= max(0, int(0.99 * target) - 5):
                                    break
                                time.sleep(0.02)
                            if vel is None:
                                time.sleep(0.5)
                            elif abs(vel) < max(0, int(0.99 * target) - 5):
                                print(f"[EPOS] velocity wait timeout: vel={vel} target={target}")
                        motor_current = epos.get_current_mA()
                        vel = epos.get_velocity_rpm()
                        print(f"[EPOS] current_mA={motor_current} vel_rpm={vel}")

                # --- Metadata per repetition
                meta = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "index": gp.get("index"),
                    "rep": rep,
                    "r_index": gp.get("r_index"),
                    "f_index": gp.get("f_index"),
                    "output_hz": output_hz,
                    "output_hz_effective": freq_hz_effective,
                    "freq_hz_effective": freq_hz_effective,
                    "output_rpm_effective": (freq_hz_effective * 60.0) if freq_hz_effective is not None else None,
                    "rpm_out_effective": (freq_hz_effective * 60.0) if freq_hz_effective is not None else None,
                    "signal_hz": gp.get("signal_hz"),
                    "decimation": gp.get("decimation"),
                    "motor_current_mA": motor_current,
                }

                # --- Pitaya acquisition per repetition (or skip)
                try:
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
                        pressure_cal_a=pressure_cal_a,
                        pressure_cal_b=pressure_cal_b,
                        voltage_cal_a=voltage_cal_a,
                        voltage_cal_b=voltage_cal_b,
                        probe_factor=probe_factor,
                        osc_r_ohm=osc_r_ohm,
                        freq_hz_override=freq_hz_effective,
                    )
                finally:
                    if use_epos and (not args.dry_run) and (d is not None):
                        assert epos is not None
                        epos.jog_stop()
                        if epos_rest_s > 0:
                            time.sleep(epos_rest_s)
                if (not res.ok) and use_pitaya and (not args.dry_run) and res.error:
                    if _is_pitaya_transient_error(res.error):
                        print(f"[Pitaya] transient error, attempting reconnect: {res.error}")
                        try:
                            if pitaya is not None:
                                pitaya.close()
                            pitaya = PitayaSCPI(host, port=port, timeout=args.timeout)
                            pitaya.open()
                            if use_epos and (not args.dry_run) and (d is not None):
                                assert epos is not None
                                motor_rpm = epos.jog_start(rpm_output=rpm_out, direction=d)
                                eff_out_rpm = abs(motor_rpm) / max(1, int(epos.cfg.gear_reduction))
                                print(f"[EPOS] effective_motor_rpm={motor_rpm} effective_output_rpm={eff_out_rpm:.2f}")
                                time.sleep(0.5)
                            try:
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
                                    pressure_cal_a=pressure_cal_a,
                                    pressure_cal_b=pressure_cal_b,
                                    voltage_cal_a=voltage_cal_a,
                                    voltage_cal_b=voltage_cal_b,
                                    probe_factor=probe_factor,
                                    osc_r_ohm=osc_r_ohm,
                                    freq_hz_override=freq_hz_effective,
                                )
                            finally:
                                if use_epos and (not args.dry_run) and (d is not None):
                                    assert epos is not None
                                    epos.jog_stop()
                                    if epos_rest_s > 0:
                                        time.sleep(epos_rest_s)
                        except Exception as e:
                            print(f"[Pitaya] reconnect failed: {e}")
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
        "calibration": {
            "pressure_cal_a": pressure_cal_a,
            "pressure_cal_b": pressure_cal_b,
            "voltage_cal_a": voltage_cal_a,
            "voltage_cal_b": voltage_cal_b,
            "pressure_v_min": float(cfg.get("pressure_v_min", 0.0)),
            "pressure_v_max": float(cfg.get("pressure_v_max", 5.0)),
            "pressure_p_min": float(cfg.get("pressure_p_min", 0.0)),
            "pressure_p_max": float(cfg.get("pressure_p_max", 2.5)),
        },
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
