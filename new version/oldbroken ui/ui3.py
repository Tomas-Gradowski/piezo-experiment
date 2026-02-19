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


# -----------------------------
# Paths
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parent
GEN = REPO_ROOT / "gen_config.py"
EXEC = REPO_ROOT / "executor.py"
GENERATED = REPO_ROOT / "generated_configs"


# -----------------------------
# Models
# -----------------------------
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
    periods: float
    repetitions: int

    pitaya_enabled: bool
    epos_enabled: bool

    # EPOS plan fields (written by generator into experiment_config.json)
    epos_direction: str
    epos_alternate: bool
    epos_rest_s: float
    epos_rpm_override: int
    epos_max_rpm: int

    # Pitaya plan fields
    signal_hz_multiplier: float


def find_latest_run_dir(base: Path, sample_name: str) -> Path:
    """
    generator creates: generated_configs/<timestamp>/<sample_name>/
    Find most recently modified matching folder.
    """
    if not base.exists():
        raise FileNotFoundError(f"generated_configs not found: {base}")

    candidates: list[Path] = []
    for p in base.glob("*/*"):
        if p.is_dir() and p.name == sample_name:
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"No run dir found for sample '{sample_name}' under {base}")

    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


# -----------------------------
# UI
# -----------------------------
class App(tk.Tk):
    """
    Rewritten UI:
      - Removes confusing checkbox soup for executor
      - Introduces a single "Test Mode" selector (radio buttons)
      - Adds explicit RPM safety limit (runtime clamp) to executor via --rpm-limit
      - Adds explicit EPOS max rpm cap passed to generator (--epos-max-rpm)
      - Keeps a manual jog panel but makes it safe by default

    Requires:
      - executor.py supports: --rpm-limit (recommended)
        If your executor doesn't have it yet, add it (see note at bottom).
      - gen_config.py supports: --epos-max-rpm and --signal_hz_multiplier (already in your code)
    """
    def _apply_styles(self) -> None:
        style = ttk.Style(self)

        # Use a theme that respects style colors better than default on many Linux setups
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        # ---- Base look ----
        style.configure("TFrame", background="#0f111a")
        style.configure("TLabel", background="#0f111a", foreground="#e6e6e6")
        style.configure("TCheckbutton", background="#0f111a", foreground="#e6e6e6")
        style.configure("TRadiobutton", background="#0f111a", foreground="#e6e6e6")

        # Group boxes / separators (ttk doesn't have true groupbox styling; labels handled above)
        style.configure("TSeparator", background="#2a2f3a")

        # Inputs
        style.configure("TEntry", fieldbackground="#151826", foreground="#e6e6e6")
        style.configure("TCombobox", fieldbackground="#151826", foreground="#e6e6e6")

        # ---- Buttons ----
        # Default button baseline
        style.configure(
            "TButton",
            padding=(12, 10),
            relief="flat",
            borderwidth=0,
            focusthickness=0,
            focuscolor="none",
        )

        def map_button(stylename: str, normal_bg: str, hover_bg: str, pressed_bg: str, fg: str = "#08140c"):
            style.configure(stylename, background=normal_bg, foreground=fg)
            style.map(
                stylename,
                background=[
                    ("disabled", "#2b3140"),
                    ("pressed", pressed_bg),
                    ("active", hover_bg),
                ],
                foreground=[
                    ("disabled", "#7a8194"),
                    ("!disabled", fg),
                ],
            )

        map_button("Run.TButton",      normal_bg="#1f8f4a", hover_bg="#24a855", pressed_bg="#1b7a40")
        map_button("Continue.TButton", normal_bg="#7ee08c", hover_bg="#8bf19a", pressed_bg="#6fd47f")
        map_button("Stop.TButton",     normal_bg="#c73737", hover_bg="#e04a4a", pressed_bg="#ab2f2f", fg="#1a0b0b")

        # Optional: make Jog look “warning but safe”
        map_button("Jog.TButton",      normal_bg="#d7b84a", hover_bg="#e8ca57", pressed_bg="#b79b3e", fg="#1a1406")
    MODES = (
        ("Plan-only (simulate, no hardware)", "plan_only"),
        ("EPOS Safe Jog (motor only)", "epos_jog"),
        ("EPOS Safe Plan (motor only)", "epos_plan"),
        ("Full Run (EPOS + Pitaya)", "full_run"),
    )

    def __init__(self) -> None:
        super().__init__()
        self.title("Piezo Experiment UI (Safe Modes)")
        self.geometry("1150x760")
        self._apply_styles()
        self.configure(background="#0f111a")

        self.log_q: queue.Queue[str] = queue.Queue()
        self.proc: subprocess.Popen | None = None
        self.worker: threading.Thread | None = None
        self.stop_flag = threading.Event()

        # ---- Inputs ----
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="x")

        def add_labeled(entry_row: int, label: str, var: tk.StringVar, width: int = 18):
            ttk.Label(frm, text=label).grid(row=entry_row, column=0, sticky="w")
            e = ttk.Entry(frm, textvariable=var, width=width)
            e.grid(row=entry_row, column=1, sticky="w", padx=(8, 30))
            return e

        def add_combo(entry_row: int, label: str, var: tk.StringVar, values: tuple[str, ...], width: int = 16):
            ttk.Label(frm, text=label).grid(row=entry_row, column=0, sticky="w")
            cb = ttk.Combobox(frm, textvariable=var, values=values, state="readonly", width=width)
            cb.grid(row=entry_row, column=1, sticky="w", padx=(8, 30))
            return cb

        # Generator inputs
        self.v_sample = tk.StringVar(value="SAFE_TEST")
        self.v_min_freq = tk.StringVar(value="4")
        self.v_max_freq = tk.StringVar(value="32")
        self.v_freq_points = tk.StringVar(value="3")
        self.v_freq_scale = tk.StringVar(value="log")

        self.v_min_r = tk.StringVar(value="1000")
        self.v_max_r = tk.StringVar(value="10000")
        self.v_r_points = tk.StringVar(value="2")
        self.v_r_scale = tk.StringVar(value="linear")

        self.v_run_subdir = tk.StringVar(value="auto")
        self.v_periods = tk.StringVar(value="10")
        self.v_repetitions = tk.StringVar(value="1")

        # Plan knobs
        self.v_signal_mult = tk.StringVar(value="1.0")

        # EPOS plan fields (generator)
        self.v_epos_direction = tk.StringVar(value="cw")
        self.v_epos_alternate = tk.BooleanVar(value=False)
        self.v_epos_rest_s = tk.StringVar(value="0.0")
        self.v_epos_rpm_override = tk.StringVar(value="0")
        self.v_epos_max_rpm = tk.StringVar(value="1000")  # plan cap

        # Hardware toggles (generator side)
        self.v_gen_pitaya_enabled = tk.BooleanVar(value=True)
        self.v_gen_epos_enabled = tk.BooleanVar(value=True)

        # Layout generator form (left)
        r = 0
        add_labeled(r, "Sample name", self.v_sample); r += 1

        add_labeled(r, "Min motor freq (Hz)", self.v_min_freq); r += 1
        add_labeled(r, "Max motor freq (Hz)", self.v_max_freq); r += 1
        add_labeled(r, "Motor points", self.v_freq_points); r += 1
        add_combo(r, "Motor scale", self.v_freq_scale, ("linear", "log")); r += 1

        add_labeled(r, "Min R (ohm)", self.v_min_r); r += 1
        add_labeled(r, "Max R (ohm)", self.v_max_r); r += 1
        add_labeled(r, "R points", self.v_r_points); r += 1
        add_combo(r, "R scale", self.v_r_scale, ("linear", "log")); r += 1

        add_labeled(r, "Periods (capture)", self.v_periods); r += 1
        add_labeled(r, "Repetitions", self.v_repetitions); r += 1

        add_labeled(r, "Run subdir", self.v_run_subdir); r += 1

        # Right-side options panel
        opt = ttk.Frame(frm)
        opt.grid(row=0, column=2, rowspan=r, sticky="n", padx=(10, 0))

        ttk.Label(opt, text="Test Mode (single selector)", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")

        self.v_mode = tk.StringVar(value="plan_only")
        for label, val in self.MODES:
            ttk.Radiobutton(opt, text=label, variable=self.v_mode, value=val, command=self._on_mode_change).pack(anchor="w")

        ttk.Separator(opt, orient="horizontal").pack(fill="x", pady=8)

        # Safety knobs
        ttk.Label(opt, text="Safety", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")

        self.v_rpm_limit = tk.StringVar(value="200")  # runtime safety clamp (executor)
        row_lim = ttk.Frame(opt)
        row_lim.pack(anchor="w", pady=(4, 0))
        ttk.Label(row_lim, text="RPM safety limit (executor)").pack(side="left")
        ttk.Entry(row_lim, textvariable=self.v_rpm_limit, width=8).pack(side="left", padx=8)

        self.v_points = tk.StringVar(value="3")
        row_pts = ttk.Frame(opt)
        row_pts.pack(anchor="w", pady=(4, 0))
        ttk.Label(row_pts, text="Plan points to run").pack(side="left")
        ttk.Entry(row_pts, textvariable=self.v_points, width=8).pack(side="left", padx=8)

        self.v_prompt_rbox = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="Prompt DR08 when resistance changes", variable=self.v_prompt_rbox).pack(anchor="w", pady=(6, 0))

        self.v_no_sleep = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="No-sleep (debug only)", variable=self.v_no_sleep).pack(anchor="w")

        ttk.Separator(opt, orient="horizontal").pack(fill="x", pady=8)

        # Generator toggles
        ttk.Label(opt, text="Generator settings", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        ttk.Checkbutton(opt, text="Write Pitaya enabled in config", variable=self.v_gen_pitaya_enabled).pack(anchor="w")
        ttk.Checkbutton(opt, text="Write EPOS enabled in config", variable=self.v_gen_epos_enabled).pack(anchor="w")

        row_sig = ttk.Frame(opt)
        row_sig.pack(anchor="w", pady=(6, 0))
        ttk.Label(row_sig, text="signal_hz_multiplier").pack(side="left")
        ttk.Entry(row_sig, textvariable=self.v_signal_mult, width=8).pack(side="left", padx=8)

        ttk.Separator(opt, orient="horizontal").pack(fill="x", pady=8)

        # EPOS plan (generator)
        ttk.Label(opt, text="EPOS plan (generator writes into config)", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")

        row_ep = ttk.Frame(opt)
        row_ep.pack(anchor="w", pady=(4, 0))
        ttk.Label(row_ep, text="Direction").pack(side="left")
        ttk.Combobox(row_ep, textvariable=self.v_epos_direction, values=("cw", "ccw"),
                     state="readonly", width=6).pack(side="left", padx=8)

        ttk.Checkbutton(opt, text="Alternate direction each repetition", variable=self.v_epos_alternate).pack(anchor="w")

        row_rest = ttk.Frame(opt)
        row_rest.pack(anchor="w", pady=(6, 0))
        ttk.Label(row_rest, text="Rest seconds").pack(side="left")
        ttk.Entry(row_rest, textvariable=self.v_epos_rest_s, width=8).pack(side="left", padx=8)

        row_rpm = ttk.Frame(opt)
        row_rpm.pack(anchor="w", pady=(6, 0))
        ttk.Label(row_rpm, text="RPM override (0=auto)").pack(side="left")
        ttk.Entry(row_rpm, textvariable=self.v_epos_rpm_override, width=8).pack(side="left", padx=8)

        row_maxrpm = ttk.Frame(opt)
        row_maxrpm.pack(anchor="w", pady=(6, 0))
        ttk.Label(row_maxrpm, text="EPOS max RPM cap (plan)").pack(side="left")
        ttk.Entry(row_maxrpm, textvariable=self.v_epos_max_rpm, width=8).pack(side="left", padx=8)

        ttk.Separator(opt, orient="horizontal").pack(fill="x", pady=8)

        # Manual jog panel (safe by default)
        ttk.Label(opt, text="Manual EPOS jog (quick check)", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        self.v_jog_dir = tk.StringVar(value="cw")
        self.v_jog_rpm = tk.StringVar(value="30")
        self.v_jog_seconds = tk.StringVar(value="1.0")

        jog_row1 = ttk.Frame(opt)
        jog_row1.pack(anchor="w", pady=(4, 0))
        ttk.Label(jog_row1, text="Dir").pack(side="left")
        ttk.Combobox(jog_row1, textvariable=self.v_jog_dir, values=("cw", "ccw"),
                     state="readonly", width=6).pack(side="left", padx=6)

        jog_row2 = ttk.Frame(opt)
        jog_row2.pack(anchor="w", pady=(4, 0))
        ttk.Label(jog_row2, text="RPM").pack(side="left")
        ttk.Entry(jog_row2, textvariable=self.v_jog_rpm, width=8).pack(side="left", padx=6)

        jog_row3 = ttk.Frame(opt)
        jog_row3.pack(anchor="w", pady=(4, 0))
        ttk.Label(jog_row3, text="Seconds").pack(side="left")
        ttk.Entry(jog_row3, textvariable=self.v_jog_seconds, width=8).pack(side="left", padx=6)

        self.btn_jog = ttk.Button(opt, text="Run Safe Jog (respects RPM limit)", command=self.on_jog, style="Jog.TButton")
        self.btn_jog.pack(anchor="w", pady=(6, 0))

        # Buttons
        btns = ttk.Frame(self, padding=(10, 0))
        btns.pack(fill="x")

        self.btn_run = ttk.Button(btns, text="Generate + Run Selected Mode", command=self.on_run, style="Run.TButton")
        self.btn_run.pack(side="left")
        self.btn_continue = ttk.Button(btns, text="Continue (send Enter)", command=self.on_continue, state="disabled", style="Continue.TButton")
        self.btn_continue.pack(side="left", padx=10)
        self.btn_stop = ttk.Button(btns, text="Stop", command=self.on_stop, state="disabled", style="Stop.TButton")
        self.btn_stop.pack(side="left", padx=10)

        # Log terminal
        term = ttk.Frame(self, padding=10)
        term.pack(fill="both", expand=True)
        self.txt = tk.Text(
            term,
            wrap="word",
            bg="#0b0d14",
            fg="#d7dce5",
            insertbackground="#d7dce5",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#2a2f3a",
            padx=10,
            pady=10,
        )
        self.txt.configure(font=("DejaVu Sans Mono", 10))        
        self.txt.pack(fill="both", expand=True)


        self.after(50, self._drain_log)
        self._on_mode_change()  # apply initial mode defaults

    # -----------------------------
    # Helpers
    # -----------------------------
    def _make_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)
        return env

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

    def on_continue(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.write("\n")
                    self.proc.stdin.flush()
                    self.log("[UI] Sent ENTER to executor\n")
            except Exception as e:
                self.log(f"[UI] Failed to send ENTER: {e}\n")

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

            if interactive and ("press Enter" in line or "then press Enter" in line):
                self.btn_continue.configure(state="normal")
                self.log("[UI] Executor is waiting. Click 'Continue (send Enter)'.\n")

        try:
            rc = self.proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            rc = -1
        finally:
            self.btn_continue.configure(state="disabled")

        return rc

    def _parse_int(self, s: str, name: str, min_v: int | None = None, max_v: int | None = None) -> int:
        try:
            v = int(str(s).strip())
        except Exception:
            raise ValueError(f"{name} must be an integer")
        if min_v is not None and v < min_v:
            raise ValueError(f"{name} must be >= {min_v}")
        if max_v is not None and v > max_v:
            raise ValueError(f"{name} must be <= {max_v}")
        return v

    def _parse_float(self, s: str, name: str, min_v: float | None = None, max_v: float | None = None) -> float:
        try:
            v = float(str(s).strip())
        except Exception:
            raise ValueError(f"{name} must be a number")
        if min_v is not None and v < min_v:
            raise ValueError(f"{name} must be >= {min_v}")
        if max_v is not None and v > max_v:
            raise ValueError(f"{name} must be <= {max_v}")
        return v

    def _on_mode_change(self) -> None:
        """
        Apply sensible defaults when switching modes.
        """
        mode = self.v_mode.get()

        if mode == "plan_only":
            # safest defaults
            self.v_gen_epos_enabled.set(True)
            self.v_gen_pitaya_enabled.set(True)
            self.v_no_sleep.set(True)
            self.v_prompt_rbox.set(True)
            if self.v_rpm_limit.get().strip() == "":
                self.v_rpm_limit.set("200")

        elif mode == "epos_jog":
            self.v_gen_epos_enabled.set(True)
            # pitaya irrelevant for jog
            self.v_no_sleep.set(True)
            self.v_prompt_rbox.set(False)
            # keep rpm limit low by default
            if self.v_rpm_limit.get().strip() == "" or int(self.v_rpm_limit.get()) > 200:
                self.v_rpm_limit.set("60")

        elif mode == "epos_plan":
            self.v_gen_epos_enabled.set(True)
            self.v_gen_pitaya_enabled.set(True)  # doesn't matter; executor will disable pitaya
            self.v_no_sleep.set(True)
            self.v_prompt_rbox.set(True)
            if self.v_rpm_limit.get().strip() == "" or int(self.v_rpm_limit.get()) > 500:
                self.v_rpm_limit.set("200")

        elif mode == "full_run":
            self.v_gen_epos_enabled.set(True)
            self.v_gen_pitaya_enabled.set(True)
            self.v_no_sleep.set(False)
            self.v_prompt_rbox.set(True)
            if self.v_rpm_limit.get().strip() == "":
                self.v_rpm_limit.set("200")

    # -----------------------------
    # Actions
    # -----------------------------
    def _send_hard_stop(self, rpm_limit: int, dry_run: bool) -> None:
        try:
            stop_cmd = [
                sys.executable, str(EXEC),
                "--run-dir", str(GENERATED),
                "--use-epos",
                "--no-pitaya",
                "--epos-jog",
                "--epos-jog-dir", "cw",
                "--epos-jog-rpm", "1",
                "--epos-jog-seconds", "0.01",
                "--rpm-limit", str(rpm_limit),
            ]

            if dry_run:
                # insert after executor path
                stop_cmd.insert(2, "--dry-run")

            env = self._make_env()

            subprocess.run(
                stop_cmd,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            )

            self.after(0, lambda: self.log("[UI] Motor stop command sent.\n"))

        except subprocess.TimeoutExpired:
            self.after(0, lambda: self.log("[UI] Hard stop timed out (2s).\n"))
        except Exception as e:
            self.after(0, lambda: self.log(f"[UI] Hard stop failed: {e}\n"))
    def on_stop(self) -> None:
        """
        Hard stop:
        1) terminate running executor
        2) send micro jog command (rpm=1, 0.01s) to force EPOS velocity reset
        """
        self.stop_flag.set()

        # 1) terminate executor process
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass

        self.log("\n[Stop requested]\n")
        ## Capture UI values BEFORE threading (Tk thread-safety)
        try:
            rpm_limit = self._parse_int(self.v_rpm_limit.get(), "RPM safety limit", 1, 1000)
        except Exception:
            rpm_limit = 60  # fallback

        # Decide if we should forward --dry-run:
        # In your UI, dry-run only happens in "plan_only" mode.
        dry_run = (self.v_mode.get() == "plan_only")

        threading.Thread(
            target=self._send_hard_stop,
            args=(rpm_limit, dry_run),
            daemon=True
        ).start()

        # UI state cleanup
        self.proc = None
        self.btn_stop.configure(state="disabled")
        self.btn_continue.configure(state="disabled")
        
           
    def on_jog(self) -> None:
        """
        Safe jog uses executor's jog mode. Always motor-only.
        Respects RPM safety limit (by clamping jog RPM to rpm_limit in UI).
        """
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Busy", "Already running.")
            return

        if not EXEC.exists():
            messagebox.showerror("Missing files", "executor.py not found in repo root.")
            return

        try:
            rpm_limit = self._parse_int(self.v_rpm_limit.get(), "RPM safety limit", 1, 8000)
            rpm = self._parse_int(self.v_jog_rpm.get(), "Jog RPM", 1, 1000)
            secs = self._parse_float(self.v_jog_seconds.get(), "Jog seconds", 0.01, 60.0)
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        # hard UI clamp for jog too
        rpm = min(rpm, rpm_limit)

        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.stop_flag.clear()

        def work():
            try:
                env = self._make_env()
                exec_cmd = [
                    sys.executable, str(EXEC),
                    "--run-dir", str(GENERATED),  # executor can search inside generated_configs
                    "--use-epos",
                    "--no-pitaya",
                    "--epos-jog",
                    "--epos-jog-dir", self.v_jog_dir.get(),
                    "--epos-jog-rpm", str(rpm),
                    "--epos-jog-seconds", str(secs),
                    "--rpm-limit", str(rpm_limit),
                ]
                # jog is ALWAYS real hardware; if you want simulate jog, use plan_only mode
                rc = self._run_cmd_stream(exec_cmd, env, interactive=False)
                self.log(f"\n[Jog finished: rc={rc}]\n")
            except Exception as e:
                self.log(f"\n[UI error] {e}\n")
            finally:
                self.proc = None
                self.stop_flag.clear()
                self.btn_run.configure(state="normal")
                self.btn_stop.configure(state="disabled")

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def on_run(self) -> None:
        """
        Generate (gen_config.py) then run executor with a single, unambiguous test mode.
        """
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Busy", "Already running.")
            return

        if not GEN.exists() or not EXEC.exists():
            messagebox.showerror("Missing files", "gen_config.py or executor.py not found in repo root.")
            return

        try:
            ga = GenArgs(
                sample_name=self.v_sample.get().strip(),
                min_freq=self._parse_float(self.v_min_freq.get(), "Min freq", 1e-6),
                max_freq=self._parse_float(self.v_max_freq.get(), "Max freq", 1e-6),
                freq_points=self._parse_int(self.v_freq_points.get(), "Freq points", 1, 10_000),
                freq_scale=self.v_freq_scale.get().strip(),

                min_r=self._parse_float(self.v_min_r.get(), "Min R", 0.1),
                max_r=self._parse_float(self.v_max_r.get(), "Max R", 0.1),
                r_points=self._parse_int(self.v_r_points.get(), "R points", 1, 10_000),
                r_scale=self.v_r_scale.get().strip(),

                run_subdir=self.v_run_subdir.get().strip(),
                periods=self._parse_float(self.v_periods.get(), "Periods", 0.01),
                repetitions=self._parse_int(self.v_repetitions.get(), "Repetitions", 1, 1_000),

                pitaya_enabled=bool(self.v_gen_pitaya_enabled.get()),
                epos_enabled=bool(self.v_gen_epos_enabled.get()),

                epos_direction=self.v_epos_direction.get(),
                epos_alternate=bool(self.v_epos_alternate.get()),
                epos_rest_s=self._parse_float(self.v_epos_rest_s.get(), "Rest seconds", 0.0, 60.0),
                epos_rpm_override=self._parse_int(self.v_epos_rpm_override.get(), "RPM override", 0, 1000),
                epos_max_rpm=self._parse_int(self.v_epos_max_rpm.get(), "EPOS max RPM cap", 1, 1000),

                signal_hz_multiplier=self._parse_float(self.v_signal_mult.get(), "signal_hz_multiplier", 0.0001, 1e9),
            )

            points_n = self._parse_int(self.v_points.get(), "Plan points", 1, 1_000_000)
            rpm_limit = self._parse_int(self.v_rpm_limit.get(), "RPM safety limit", 1, 1000)
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        if ga.max_freq < ga.min_freq:
            messagebox.showerror("Invalid input", "Max freq must be >= Min freq.")
            return
        if ga.max_r < ga.min_r:
            messagebox.showerror("Invalid input", "Max R must be >= Min R.")
            return

        mode = self.v_mode.get()

        # In safe modes, force low limits unless user explicitly changed
        if mode in ("plan_only", "epos_plan", "epos_jog") and rpm_limit > 1000:
            if not messagebox.askyesno("Safety check", f"RPM safety limit is {rpm_limit}. Continue?"):
                return

        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.stop_flag.clear()

        def work():
            try:
                env = self._make_env()

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
                    "--periods", str(ga.periods),
                    "--repetitions", str(ga.repetitions),

                    "--signal_hz_multiplier", str(ga.signal_hz_multiplier),
                ]

                if ga.pitaya_enabled:
                    gen_cmd.append("--pitaya-enabled")

                if ga.epos_enabled:
                    gen_cmd += [
                        "--epos-enabled",
                        "--epos-direction", ga.epos_direction,
                        "--epos-max-rpm", str(ga.epos_max_rpm),
                        "--epos-rest-s", str(ga.epos_rest_s),
                        "--epos-rpm-override", str(ga.epos_rpm_override),
                    ]
                    if ga.epos_alternate:
                        gen_cmd.append("--epos-alternate")

                rc = self._run_cmd_stream(gen_cmd, env)
                if self.stop_flag.is_set():
                    self.log("\n[Stopped]\n")
                    return
                if rc != 0:
                    self.log(f"\n[Generator failed: rc={rc}]\n")
                    return

                time.sleep(0.2)

                # 2) find latest run dir
                run_dir = find_latest_run_dir(GENERATED, ga.sample_name)
                self.log(f"\n[Using run-dir: {run_dir}]\n")

                # 3) run executor based on mode
                exec_cmd = [sys.executable, str(EXEC), "--run-dir", str(run_dir)]

                # Always apply rpm safety limit when we might touch EPOS
                # (plan_only is dry-run so it doesn't connect anyway, but clamp still documents intent)
                exec_cmd += ["--rpm-limit", str(rpm_limit)]

                # DR08 prompting is useful in all modes that run plan loop
                if self.v_prompt_rbox.get() and mode in ("plan_only", "epos_plan", "full_run"):
                    exec_cmd.append("--prompt-rbox")

                if self.v_no_sleep.get():
                    exec_cmd.append("--no-sleep")

                # Mode mapping
                if mode == "plan_only":
                    # simulate everything, no hardware connections
                    exec_cmd += ["--dry-run", "--use-epos", "--no-pitaya", "--points", str(points_n)]
                    interactive = True

                elif mode == "epos_plan":
                    # real motor moves, but no pitaya acquisition
                    exec_cmd += ["--use-epos", "--no-pitaya", "--points", str(points_n)]
                    interactive = True

                elif mode == "full_run":
                    # real motor + real pitaya acquisition
                    exec_cmd += ["--use-epos", "--points", str(points_n)]
                    interactive = True

                else:
                    # Shouldn't happen; jog has its own button
                    self.log("\n[UI] Unknown mode for Run. Use the Jog button for EPOS Safe Jog.\n")
                    return

                rc2 = self._run_cmd_stream(exec_cmd, env, interactive=interactive)
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
                self.btn_continue.configure(state="disabled")

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()


# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    app = App()
    app.mainloop()

"""
NOTE: executor.py must support --rpm-limit for this UI.

Minimal patch for executor.py:

1) add arg:
    ap.add_argument("--rpm-limit", type=int, default=1000,
                    help="Hard safety cap on commanded rpm (runtime clamp).")

2) clamp where rpm_out computed in EPOS loop:
    rpm_out = epos_rpm_override if epos_rpm_override > 0 else rpm_out_plan
    rpm_out = max(0, min(rpm_out, int(args.rpm_limit)))

3) also clamp manual jog rpm before calling epos.jog_start:
    jog_rpm = max(0, min(args.epos_jog_rpm, int(args.rpm_limit)))

This makes it impossible to ever command > rpm-limit from UI or CLI.
"""
