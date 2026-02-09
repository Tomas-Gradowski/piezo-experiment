#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# SCPI client
# -----------------------------

class SCPIClient:
    """
    Minimal SCPI over TCP client for Red Pitaya.
    - send(cmd): writes a command (no response expected)
    - query(cmd): writes and reads until newline (response expected)
    """
    def __init__(self, host: str, port: int, timeout_s: float = 10.0):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.sock: Optional[socket.socket] = None

    def connect(self) -> None:
        s = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        s.settimeout(self.timeout_s)
        self.sock = s

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def send(self, cmd: str) -> None:
        if not self.sock:
            raise RuntimeError("SCPI socket not connected")
        payload = (cmd.strip() + "\n").encode("utf-8")
        self.sock.sendall(payload)

    def query(self, cmd: str) -> str:
        self.send(cmd)
        return self._readline()

    def _readline(self) -> str:
        if not self.sock:
            raise RuntimeError("SCPI socket not connected")
        data = bytearray()
        while True:
            chunk = self.sock.recv(4096)
            if not chunk:
                break
            data.extend(chunk)
            if b"}" in chunk or chunk.endswith(b"\n"):
                break

            # safety cap (prevents infinite loop if device misbehaves)
            if len(data) > 50_000_000:
                raise RuntimeError("SCPI response too large (possible read hang)")

        return data.decode("utf-8", errors="replace").strip()


# -----------------------------
# Parsing helpers
# -----------------------------

def parse_waveform_text(resp: str) -> List[float]:
    """
    Red Pitaya SCPI responses often look like:
      "{0.001,0.002,...}"
    or can be comma-separated without braces.
    We handle:
      - braces {}
      - commas and/or spaces
      - empty tokens
    """
    s = resp.strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()

    # split by comma first; if no commas, split by whitespace
    parts = s.split(",") if "," in s else s.split()
    out: List[float] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Sometimes RP can return "nan" or "inf"; keep as float if possible
        try:
            out.append(float(p))
        except ValueError:
            # Last resort: try to strip weird chars
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
    ch1: List[float],
    ch2: List[float],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "ch1", "ch2"])
        for i in range(min(len(t), len(ch1), len(ch2))):
            w.writerow([t[i], ch1[i], ch2[i]])


# -----------------------------
# Executor core
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
    scpi: SCPIClient,
    point: Dict[str, Any],
    fs_hz: float,
    n_samples: int,
    base_out: Path,
    no_sleep: bool = False,
    extra_wait_s: float = 0.1,
    dry_run: bool = False,
) -> ExecResult:
    idx = int(point["index"])
    dec = int(point["decimation"])

    out_csv = base_out / "csv" / f"point_{idx:06d}.csv"

    if dry_run:
        return ExecResult(
            index=idx, ok=True, error=None, out_csv=str(out_csv), n_ch1=n_samples, n_ch2=n_samples
        )

    try:
        # setup
        for cmd in point["pitaya_setup"]:
            scpi.send(cmd)

        # wait for acquisition to complete (C#-style)
        meas_time = (dec / fs_hz) * n_samples + extra_wait_s
        if not no_sleep:
            time.sleep(meas_time)

        # query waveforms
        ch1_txt = scpi.query(point["pitaya_queries"][0])
        ch2_txt = scpi.query(point["pitaya_queries"][1])

        ch1 = parse_waveform_text(ch1_txt)
        ch2 = parse_waveform_text(ch2_txt)

        # time axis
        t = make_time_axis(n_samples=n_samples, fs_hz=fs_hz, decimation=dec)

        # save
        save_csv_point(out_csv, t=t, ch1=ch1, ch2=ch2)

        return ExecResult(
            index=idx, ok=True, error=None, out_csv=str(out_csv), n_ch1=len(ch1), n_ch2=len(ch2)
        )

    except Exception as e:
        return ExecResult(index=idx, ok=False, error=str(e), out_csv=None)


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Execute measurement_plan.json on Red Pitaya and export CSV waveforms.")
    ap.add_argument("--run-dir", required=True,
                    help="Folder created by generator (contains experiment_config.json + measurement_plan.json)")
    ap.add_argument("--host", default=None, help="Override pitaya host (otherwise from experiment_config.json)")
    ap.add_argument("--port", type=int, default=None, help="Override pitaya port (otherwise from experiment_config.json)")
    ap.add_argument("--timeout", type=float, default=10.0, help="TCP timeout seconds")
    ap.add_argument("--dry-run", action="store_true", help="Do not connect or run; just show what would happen.")
    ap.add_argument("--no-sleep", action="store_true", help="Skip waiting for acquisition (debug only).")
    ap.add_argument("--extra-wait", type=float, default=0.1, help="Extra wait seconds after acquisition time.")
    ap.add_argument("--start-index", type=int, default=0, help="Start grid point index (inclusive).")
    ap.add_argument("--stop-index", type=int, default=None, help="Stop grid point index (exclusive).")
    ap.add_argument("--points", type=int, default=None, help="Run only first N points from start-index.")
    ap.add_argument("--fail-fast", action="store_true", help="Stop at first failure.")
    return ap.parse_args()

def main() -> None:
    args = parse_args()

    run_dir_in = Path(args.run_dir).expanduser().resolve()

    def resolve_run_dir(p: Path) -> Path:
        # If user passed a file, use its parent
        if p.is_file():
            p = p.parent

        cfg_name = "experiment_config.json"
        plan_name = "measurement_plan.json"

        # If JSONs are directly inside p, done.
        if (p / cfg_name).exists() and (p / plan_name).exists():
            return p

        if not p.is_dir():
            raise FileNotFoundError(f"{p} is not a directory and does not contain {cfg_name}/{plan_name}")

        # Search recursively for directories containing both files
        matches: List[Path] = []
        for cfg_path in p.rglob(cfg_name):
            cand = cfg_path.parent
            if (cand / plan_name).exists():
                matches.append(cand)

        if len(matches) == 1:
            return matches[0]

        if len(matches) == 0:
            raise FileNotFoundError(
                f"Could not find {cfg_name} + {plan_name} anywhere under {p}."
            )

        # If there are multiple, choose the most recently modified one (nice default),
        # but also print options so you can be explicit.
        matches.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        chosen = matches[0]
        print("Warning: multiple run folders found. Using most recent:")
        for m in matches[:10]:
            print(f"  - {m}")
        return chosen
    run_dir = resolve_run_dir(run_dir_in)
    cfg_path = run_dir / "experiment_config.json"
    plan_path = run_dir / "measurement_plan.json"

    cfg = load_json(cfg_path)
    plan = load_json(plan_path)
    pitaya_cfg = cfg["pitaya"]
    host = args.host if args.host is not None else pitaya_cfg["host"]
    port = args.port if args.port is not None else int(pitaya_cfg["port"])
    fs_hz = float(pitaya_cfg["fs_hz"])
    n_samples = int(pitaya_cfg["n_samples"])

    points: List[Dict[str, Any]] = plan["grid_points"]

    start = max(0, int(args.start_index))
    stop = int(args.stop_index) if args.stop_index is not None else len(points)
    stop = min(stop, len(points))
    subset = points[start:stop]

    if args.points is not None:
        subset = subset[: max(0, int(args.points))]

    # output base: same run folder, so analysis scripts can just point to it
    base_out = run_dir

    print(f"Run dir: {run_dir}")
    print(f"Pitaya: {host}:{port} (timeout={args.timeout}s)")
    print(f"Samples: n_samples={n_samples}, fs_hz={fs_hz}")
    print(f"Executing points: {len(subset)} (indexes {start}..{start + len(subset) - 1})")
    print(f"CSV out: {base_out / 'csv'}")
    if args.dry_run:
        print("Dry-run: will not connect / execute.")
        # Still show first point commands to sanity-check
        if subset:
            p0 = subset[0]
            print("\nFirst point preview:")
            print(f"  index={p0['index']} f={p0['freq_hz']}Hz R={p0['load_r_ohm']}Ω dec={p0['decimation']}")
            for c in p0["pitaya_setup"]:
                print(f"    {c}")
            for q in p0["pitaya_queries"]:
                print(f"    {q}")
        return

    scpi = SCPIClient(host=host, port=port, timeout_s=args.timeout)
    results: List[ExecResult] = []
    t0 = time.time()

    try:
        scpi.connect()
        for p in subset:
            res = execute_point(
                scpi=scpi,
                point=p,
                fs_hz=fs_hz,
                n_samples=n_samples,
                base_out=base_out,
                no_sleep=args.no_sleep,
                extra_wait_s=args.extra_wait,
                dry_run=False,
            )
            results.append(res)
            if res.ok:
                print(f"[OK] point {res.index} -> {res.out_csv} (n1={res.n_ch1}, n2={res.n_ch2})")
            else:
                print(f"[FAIL] point {res.index}: {res.error}")
                if args.fail_fast:
                    break

    finally:
        scpi.close()

    dt = time.time() - t0

    # manifest
    manifest = {
        "run_dir": str(run_dir),
        "pitaya": {"host": host, "port": port},
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
