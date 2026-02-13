#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import queue
import threading
import subprocess
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox


REPO_ROOT = Path(__file__).resolve().parent
GEN = REPO_ROOT / "gen_config.py"
EXEC = REPO_ROOT / "executor.py"
GENERATED = REPO_ROOT / "generated_configs"


@dataclass
class GenArgs:
    sample_name: str
    min_freq: float
    max_freq: float
    freq_points: int
    freq_scale: str
    min_r: float
    max_r: float
    r_points: int
    r_scale: str
    run_subdir: str


def find_latest_run_dir(base: Path, sample_name: str) -> Path:
    """
    Your generator creates: generated_configs/<timestamp>/<sample_name>/
    We find the most recently modified matching folder.
    """
    if not base.exists():
        raise FileNotFoundError(f"generated_configs not found: {base}")

    candidates = []
    for p in base.glob("*/*"):
        if p.is_dir() and p.name == sample_name:
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"No run dir found for sample '{sample_name}' under {base}")

    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Piezo Experiment UI (gen_config → executor)")
        self.geometry("980x640")

        self.log_q: queue.Queue[str] = queue.Queue()
        self.proc: subprocess.Popen | None = None
        self.worker: threading.Thread | None = None
        self.stop_flag = threading.Event()

        # ---- Inputs ----
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="x")

        def add_labeled(entry_row, label, var, width=18):
            ttk.Label(frm, text=label).grid(row=entry_row, column=0, sticky="w")
            e = ttk.Entry(frm, textvariable=var, width=width)
            e.grid(row=entry_row, column=1, sticky="w", padx=(8, 30))
            return e
        def add_combo(entry_row, label, var, values):
            ttk.Label(frm, text=label).grid(row=entry_row, column=0, sticky="w")
            cb = ttk.Combobox(frm, textvariable=var, values=values, state="readonly", width=16)
            cb.grid(row=entry_row, column=1, sticky="w", padx=(8, 30))
            return cb
        self.v_sample = tk.StringVar(value="test N")
        self.v_min_freq = tk.StringVar(value=">=10")
        self.v_max_freq = tk.StringVar(value="=<1000")
        self.v_freq_points = tk.StringVar(value="3")
        self.v_freq_scale = tk.StringVar(value="log")

        self.v_min_r = tk.StringVar(value=">=1000")
        self.v_max_r = tk.StringVar(value="=<10000")
        self.v_r_points = tk.StringVar(value="2")
        self.v_r_scale = tk.StringVar(value="linear")

        self.v_run_subdir = tk.StringVar(value="auto")

        r = 0
        add_labeled(r, "Sample name", self.v_sample); r += 1

        add_labeled(r, "Min freq (Hz)", self.v_min_freq); r += 1
        add_labeled(r, "Max freq (Hz)", self.v_max_freq); r += 1
        add_labeled(r, "Freq points", self.v_freq_points); r += 1
        add_combo(r, "Freq scale", self.v_freq_scale, ("linear", "log")); r += 1

        add_labeled(r, "Min R (ohm)", self.v_min_r); r += 1
        add_labeled(r, "Max R (ohm)", self.v_max_r); r += 1
        add_labeled(r, "R points", self.v_r_points); r += 1
        add_combo(r, "R scale", self.v_r_scale, ("linear", "log")); r += 1

        add_labeled(r, "Run subdir", self.v_run_subdir); r += 1
        # keep min/max/points as entries
        
                # Executor toggles
        opt = ttk.Frame(frm)
        opt.grid(row=0, column=2, rowspan=r, sticky="n", padx=(10, 0))
        ttk.Label(opt, text="Executor options").pack(anchor="w")

        self.v_dry_run = tk.BooleanVar(value=True)
        self.v_prompt_rbox = tk.BooleanVar(value=True)
        self.v_no_sleep = tk.BooleanVar(value=True)
        self.v_points = tk.StringVar(value="5")
        self.v_use_epos = tk.BooleanVar(value=False)

        ttk.Checkbutton(opt, text="--dry-run", variable=self.v_dry_run).pack(anchor="w")
        ttk.Checkbutton(opt, text="--prompt-rbox", variable=self.v_prompt_rbox).pack(anchor="w")
        ttk.Checkbutton(opt, text="--no-sleep", variable=self.v_no_sleep).pack(anchor="w")
        ttk.Checkbutton(opt, text="--use-epos", variable=self.v_use_epos).pack(anchor="w")
        ttk.Label(opt, text="--points").pack(anchor="w", pady=(10, 0))
        ttk.Entry(opt, textvariable=self.v_points, width=8).pack(anchor="w")

        # Buttons
        btns = ttk.Frame(self, padding=(10, 0))
        btns.pack(fill="x")

        self.btn_run = ttk.Button(btns, text="Generate config + Start experiment", command=self.on_run)
        self.btn_run.pack(side="left")
        self.btn_continue = ttk.Button(btns, text="Continue (send Enter)", command=self.on_continue, state="disabled")
        self.btn_continue.pack(side="left", padx=10)
        self.btn_stop = ttk.Button(btns, text="Stop", command=self.on_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=10)

        # Log terminal
        term = ttk.Frame(self, padding=10)
        term.pack(fill="both", expand=True)

        self.txt = tk.Text(term, wrap="word")
        self.txt.pack(fill="both", expand=True)
        self.txt.configure(font=("Monospace", 10))

        self.after(50, self._drain_log)
    def on_continue(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.write("\n")
                    self.proc.stdin.flush()
                    self.log("[UI] Sent ENTER to executor\n")
            except Exception as e:
                self.log(f"[UI] Failed to send ENTER: {e}\n")
    def log(self, s: str) -> None:
        self.log_q.put(s)

    def _drain_log(self) -> None:
        try:
            while True:
                s = self.log_q.get_nowait()
                self.txt.insert("end", s)
                self.txt.see("end")
        except queue.Empty:
            pass
        self.after(50, self._drain_log)
    def _run_cmd_stream(self, cmd: list[str], env: dict[str, str], interactive: bool = False) -> int:
        self.log(f"\n$ {' '.join(cmd)}\n")
        self.proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdin=subprocess.PIPE if interactive else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
        assert self.proc.stdout is not None

    
        if interactive:
            self.btn_continue.configure(state="disabled")

        for line in self.proc.stdout:
            if self.stop_flag.is_set():
                break
            self.log(line)

            # Detect executor waiting for user input
            if interactive and ("press Enter" in line or "then press Enter" in line):
                self.btn_continue.configure(state="normal")
                self.log("[UI] Executor is waiting. Click 'Continue (send Enter)'.\n")

        try:
            rc = self.proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            rc = -1
        finally:
            # Disable continue when process ends
            self.btn_continue.configure(state="disabled")

        return rc
    

    def on_run(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Busy", "Already running.")
            return

        # Validate inputs quickly
        try:
            ga = GenArgs(
                sample_name=self.v_sample.get().strip(),
                min_freq=float(self.v_min_freq.get()),
                max_freq=float(self.v_max_freq.get()),
                freq_points=int(self.v_freq_points.get()),
                freq_scale=self.v_freq_scale.get().strip(),
                min_r=float(self.v_min_r.get()),
                max_r=float(self.v_max_r.get()),
                r_points=int(self.v_r_points.get()),
                r_scale=self.v_r_scale.get().strip(),
                run_subdir=self.v_run_subdir.get().strip(),
            )
            points_n = int(self.v_points.get())
            if points_n < 1:
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid input", "Check numeric fields (freq/r/points).")
            return

        if not GEN.exists() or not EXEC.exists():
            messagebox.showerror("Missing files", "gen_config.py or executor.py not found in repo root.")
            return

        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.stop_flag.clear()

        def work():
            try:
                # Ensure local imports work (devices package)
                env = os.environ.copy()
                env["PYTHONPATH"] = str(REPO_ROOT)

                # 1) run generator
                gen_cmd = [
                    sys.executable, str(GEN),
                    "--sample-name", ga.sample_name,
                    "--min-freq", str(ga.min_freq),
                    "--max-freq", str(ga.max_freq),
                    "--freq-points", str(ga.freq_points),
                    "--freq-scale", ga.freq_scale,
                    "--min-r", str(ga.min_r),
                    "--max-r", str(ga.max_r),
                    "--r-points", str(ga.r_points),
                    "--r-scale", ga.r_scale,
                    "--run-subdir", ga.run_subdir,
                ]
                rc = self._run_cmd_stream(gen_cmd, env)
                if self.stop_flag.is_set():
                    self.log("\n[Stopped]\n")
                    return
                if rc != 0:
                    self.log(f"\n[Generator failed: rc={rc}]\n")
                    return

                # small delay to allow filesystem timestamps to update
                time.sleep(0.2)

                # 2) find latest run dir
                run_dir = find_latest_run_dir(GENERATED, ga.sample_name)
                self.log(f"\n[Using run-dir: {run_dir}]\n")

                # 3) run executor
                exec_cmd = [sys.executable, str(EXEC), "--run-dir", str(run_dir)]
                if self.v_prompt_rbox.get():
                    exec_cmd.append("--prompt-rbox")
                if self.v_dry_run.get():
                    exec_cmd.append("--dry-run")
                if self.v_no_sleep.get():
                    exec_cmd.append("--no-sleep")
                if self.v_use_epos.get():
                    exec_cmd.append("--use-epos")
                exec_cmd += ["--points", self.v_points.get()]

                rc2 = self._run_cmd_stream(exec_cmd, env,interactive=True)
                if self.stop_flag.is_set():
                    self.log("\n[Stopped]\n")
                    return
                self.log(f"\n[Executor finished: rc={rc2}]\n")

            except Exception as e:
                self.log(f"\n[UI error] {e}\n")
            finally:
                self.proc = None
                self.stop_flag.clear()
                self.btn_run.configure(state="normal")
                self.btn_stop.configure(state="disabled")

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def on_stop(self) -> None:
        self.stop_flag.set()
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.log("\n[Stop requested]\n")


if __name__ == "__main__":
    app = App()
    app.mainloop()
