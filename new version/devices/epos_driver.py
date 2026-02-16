#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from ctypes import cdll,CDLL, byref, c_uint, c_short, c_int
import time
from typing import Optional


class Direction(str, Enum):
    CW = "cw"
    CCW = "ccw"


@dataclass
class EposConfig:
    # EPOS library + connection
    lib_path: str
    node_id: int = 1
    device_name: bytes = b"EPOS4"
    protocol_stack: bytes = b"MAXON SERIAL V2"
    interface: bytes = b"USB"
    port: bytes = b"USB0"

    # Mechanics
    gear_reduction: int = 18          # motor_rpm = output_rpm * gear_reduction
    one_turn: int = 73728             # increments per mechanical turn (your constant)
    max_rpm: int = 8000               # motor max rpm (after gear multiplication)

    # Motion defaults
    accel: int = 10000
    decel: int = 10000


class EposDriver:
    """
    Minimal, safe EPOS4 wrapper.

    Key points:
    - NO library load at import time (so executor --dry-run won't crash)
    - open() loads DLL and opens device
    - provides:
        - jog_start/jog_stop (CW/CCW testing)
        - move_for_time (velocity mode)
        - move_n_turns / move_to_position (position mode)
        - run_cycles(freq_hz, periods, reps, ...) high-level helper for your experiment
    """

    def __init__(self, cfg: EposConfig):
        self.cfg = cfg
        self.epos: Optional[CDLL] = None
        self.keyhandle: int = 0
        self.pErrorCode = c_uint(0)
        self.pDeviceErrorCode = c_uint(0)

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _require_open(self) -> None:
        if self.epos is None or self.keyhandle == 0:
            raise RuntimeError("EPOS not opened. Call epos.open() first.")

    def _clamp_motor_rpm(self, motor_rpm: int) -> int:
        motor_rpm = int(motor_rpm)
        if motor_rpm > self.cfg.max_rpm:
            motor_rpm = self.cfg.max_rpm
        if motor_rpm < -self.cfg.max_rpm:
            motor_rpm = -self.cfg.max_rpm
        return motor_rpm

    def _dir_sign(self, direction: Direction) -> int:
        # Your legacy main.py inverts sign for CW in velocity mode. :contentReference[oaicite:1]{index=1}
        return -1 if direction == Direction.CW else 1

    # -----------------------------
    # Lifecycle
    # -----------------------------
    def open(self) -> None:
        # Lazy-load library only here
        self.epos = CDLL(self.cfg.lib_path)

        self.keyhandle = self.epos.VCS_OpenDevice(
            self.cfg.device_name,
            self.cfg.protocol_stack,
            self.cfg.interface,
            self.cfg.port,
            byref(self.pErrorCode),
        )
        if self.keyhandle == 0:
            raise RuntimeError(f"VCS_OpenDevice failed (err={self.pErrorCode.value})")

        # Check device error code (pattern from your old GUI) :contentReference[oaicite:2]{index=2}
        ret = self.epos.VCS_GetDeviceErrorCode(
            self.keyhandle,
            self.cfg.node_id,
            1,
            byref(self.pDeviceErrorCode),
            byref(self.pErrorCode),
        )
        if ret == 0:
            raise RuntimeError(f"VCS_GetDeviceErrorCode failed (err={self.pErrorCode.value})")

        if self.pDeviceErrorCode.value != 0:
            raise RuntimeError(
                f"EPOS device error={self.pDeviceErrorCode.value} (err={self.pErrorCode.value})"
            )

        # Enable
        self.epos.VCS_SetEnableState(self.keyhandle, self.cfg.node_id, byref(self.pErrorCode))

    def close(self) -> None:
        if self.epos is not None and self.keyhandle != 0:
            self.epos.VCS_CloseDevice(self.keyhandle, byref(self.pErrorCode))
        self.keyhandle = 0
        self.epos = None

    # -----------------------------
    # Telemetry
    # -----------------------------
    def get_current_mA(self) -> int:
        self._require_open()
        cur = c_short()
        self.epos.VCS_GetCurrentIsAveraged(
            self.keyhandle, self.cfg.node_id, byref(cur), byref(self.pErrorCode)
        )
        return int(cur.value)

    def get_velocity_rpm(self) -> Optional[int]:
        """
        Tries to read averaged velocity. Not all builds expose this symbol.
        Returns None if function not available.
        """
        self._require_open()
        if not hasattr(self.epos, "VCS_GetVelocityIsAveraged"):
            return None
        vel = c_int()
        ok = self.epos.VCS_GetVelocityIsAveraged(
            self.keyhandle, self.cfg.node_id, byref(vel), byref(self.pErrorCode)
        )
        if ok == 0:
            return None
        return int(vel.value)

    # -----------------------------
    # Low-level motion (Velocity mode)
    # -----------------------------
    def jog_start(self, rpm_output: int, direction: Direction = Direction.CW) -> int:
        """
        Start spinning continuously (manual test).
        rpm_output is output-shaft rpm. We'll convert to motor rpm via gear reduction.
        """
        self._require_open()
        self.epos.VCS_ActivateProfileVelocityMode(self.keyhandle, self.cfg.node_id, byref(self.pErrorCode))

        motor_rpm = int(rpm_output) * int(self.cfg.gear_reduction)
        motor_rpm = self._clamp_motor_rpm(motor_rpm)
        motor_rpm *= self._dir_sign(direction)

        self.epos.VCS_MoveWithVelocity(self.keyhandle, self.cfg.node_id, int(motor_rpm), byref(self.pErrorCode))
        return motor_rpm

    def jog_stop(self) -> None:
        self._require_open()
        self.epos.VCS_HaltVelocityMovement(self.keyhandle, self.cfg.node_id, byref(self.pErrorCode))

    def move_for_time(self, duration_s: float, rpm_output: int, direction: Direction = Direction.CW) -> int:
        """
        Spin for duration_s seconds at rpm_output (output rpm), then stop.
        Matches your old move_for_time logic. :contentReference[oaicite:3]{index=3}
        """
        self._require_open()
        motor_rpm = self.jog_start(rpm_output=rpm_output, direction=direction)
        t0 = time.time()
        while (time.time() - t0) < float(duration_s):
            # keep commanding (like your old loop)
            self.epos.VCS_MoveWithVelocity(self.keyhandle, self.cfg.node_id, int(motor_rpm), byref(self.pErrorCode))
            time.sleep(0.01)
        self.jog_stop()
        return motor_rpm

    # -----------------------------
    # Low-level motion (Position mode)
    # -----------------------------
    def move_to_position_increments(self, target_increments: int, rpm_output: int, direction: Direction = Direction.CW) -> None:
        """
        Move to target position in increments (internal units) using profile position mode.
        """
        self._require_open()
        self.epos.VCS_ActivateProfilePositionMode(self.keyhandle, self.cfg.node_id, byref(self.pErrorCode))

        motor_rpm = int(rpm_output) * int(self.cfg.gear_reduction)
        motor_rpm = abs(self._clamp_motor_rpm(motor_rpm))

        self.epos.VCS_SetPositionProfile(
            self.keyhandle,
            self.cfg.node_id,
            int(motor_rpm),
            int(self.cfg.accel),
            int(self.cfg.decel),
            byref(self.pErrorCode),
        )

        # Your legacy code flips sign for CW in position mode too. :contentReference[oaicite:4]{index=4}
        if direction == Direction.CW:
            target_increments *= -1

        self.epos.VCS_MoveToPosition(
            self.keyhandle,
            self.cfg.node_id,
            int(target_increments),
            0,  # absolute
            0,  # immediately
            byref(self.pErrorCode),
        )

    def move_n_turns(self, turns: float, rpm_output: int, direction: Direction = Direction.CW) -> None:
        """
        Move N turns (relative) based on your one_turn constant.
        """
        inc = int(float(turns) * int(self.cfg.one_turn))
        self.move_to_position_increments(inc, rpm_output=rpm_output, direction=direction)

    # -----------------------------
    # High-level program for your experiment
    # -----------------------------
    def stop(self) -> None:
        self.jog_stop()
    def close(self) -> None:
        if self.epos is not None and self.keyhandle != 0:
            try:
                self.epos.VCS_HaltVelocityMovement(self.keyhandle, self.cfg.node_id, byref(self.pErrorCode))
            except Exception:
                pass
        self.epos.VCS_CloseDevice(self.keyhandle, byref(self.pErrorCode))
        self.keyhandle = 0
        self.epos = None
    def run_cycles(
        self,
        freq_hz: float,
        periods: int,
        repetitions: int,
        rpm_output: Optional[int] = None,
        direction: Direction | str = Direction.CW,
        rest_s: float = 0.0,
        alternate: bool = False,
    ) -> None:
        """
        "frequency / periods / repetitions" runner.

        Interpretation:
          duration_s = periods / freq_hz

        rpm_output:
          - if provided: use it
          - else: derive from freq_hz assuming freq_hz = rotations per second (output shaft):
                output_rpm = 60 * freq_hz

        direction:
          - cw or ccw
        alternate:
          - if True: rep 0 uses direction, rep 1 uses opposite, etc.
        """
        self._require_open()

        freq_hz = float(freq_hz)
        if freq_hz <= 0:
            raise ValueError("freq_hz must be > 0")
        if periods <= 0 or repetitions <= 0:
            raise ValueError("periods and repetitions must be > 0")

        dur = float(periods) / freq_hz

        base_dir = Direction(direction) if isinstance(direction, str) else direction
        for rep in range(int(repetitions)):
            d = base_dir
            if alternate and (rep % 2 == 1):
                d = Direction.CCW if base_dir == Direction.CW else Direction.CW

            if rpm_output is None:
                rpm = int(round(60.0 * freq_hz))
            else:
                rpm = int(rpm_output)

            self.move_for_time(duration_s=dur, rpm_output=rpm, direction=d)

            if rest_s > 0:
                time.sleep(float(rest_s))       
            self.epos.VCS_HaltVelocityMovement(self.keyhandle, self.cfg.node_id, byref(self.pErrorCode))
