from __future__ import annotations
from ctypes import cdll,CDLL, byref, c_uint, c_short
from dataclasses import dataclass
import time

path1 = '/home/tomas/thesis/epos/EPOS-Linux-Library-En/EPOS_Linux_Library/lib/intel/x86_64/libEposCmd.so.6.8.1.0'
#path1 = '/home/pi/EPOS_Linux_Library/lib/arm/v7/libEposCmd.so.6.7.1.0'

cdll.LoadLibrary(path1)
epos = CDLL(path1)
# How many increments is one turn
one_turn = 73728

# Node ID must match with Hardware Dip-Switch setting of EPOS4
NodeID = 1
keyhandle = 0
# return variable from Library Functions
ret = 0
pErrorCode = c_uint()
pDeviceErrorCode = c_uint()

@dataclass
class EposConfig:
    lib_path: str = path1
    node_id: int = 1
    device_name: bytes = b'EPOS4'
    protocol_stack: bytes = b'MAXON SERIAL V2'
    interface: bytes = b'USB'
    port: bytes = b'USB0'
    gear_reduction: int = 18
    one_turn: int = 73728
    max_rpm: int = 8000

class EposDriver:
    def __init__(self, cfg: EposConfig):
        self.cfg = cfg
        self.epos = CDLL(cfg.lib_path)
        self.keyhandle = 0
        self.pErrorCode = c_uint()
        self.pDeviceErrorCode = c_uint()

    def open(self) -> None:
        self.keyhandle = self.epos.VCS_OpenDevice(
            self.cfg.device_name,
            self.cfg.protocol_stack,
            self.cfg.interface,
            self.cfg.port,
            byref(self.pErrorCode),
        )
        if self.keyhandle == 0:
            raise RuntimeError(f"VCS_OpenDevice failed (err={self.pErrorCode.value})")

        ret = self.epos.VCS_GetDeviceErrorCode(
            self.keyhandle, self.cfg.node_id, 1,
            byref(self.pDeviceErrorCode), byref(self.pErrorCode)
        )
        if self.pDeviceErrorCode.value != 0:
            raise RuntimeError(f"EPOS device error={self.pDeviceErrorCode.value} (err={self.pErrorCode.value})")

        self.epos.VCS_SetEnableState(self.keyhandle, self.cfg.node_id, byref(self.pErrorCode))

    def close(self) -> None:
        if self.keyhandle:
            self.epos.VCS_CloseDevice(self.keyhandle, byref(self.pErrorCode))
            self.keyhandle = 0

    def get_current_mA(self) -> int:
        cur = c_short()
        self.epos.VCS_GetCurrentIsAveraged(self.keyhandle, self.cfg.node_id, byref(cur), byref(self.pErrorCode))
        return int(cur.value)

    # examples you can call from executor
    def move_to_position(self, target_position: int, velocity_rpm: int, cw: bool = True) -> None:
        self.epos.VCS_ActivateProfilePositionMode(self.keyhandle, self.cfg.node_id, byref(self.pErrorCode))
        vel = min(int(velocity_rpm), self.cfg.max_rpm)
        self.epos.VCS_SetPositionProfile(self.keyhandle, self.cfg.node_id, vel, 10000, 10000, byref(self.pErrorCode))

        if cw:
            target_position *= -1
        self.epos.VCS_MoveToPosition(self.keyhandle, self.cfg.node_id, int(target_position), 0, 0, byref(self.pErrorCode))

    def move_for_time(self, duration_s: float, velocity_rpm: int, cw: bool = True) -> None:
        self.epos.VCS_ActivateProfileVelocityMode(self.keyhandle, self.cfg.node_id, byref(self.pErrorCode))
        vel = min(int(velocity_rpm), self.cfg.max_rpm)
        if cw:
            vel *= -1

        t0 = time.time()
        while time.time() - t0 < duration_s:
            self.epos.VCS_MoveWithVelocity(self.keyhandle, self.cfg.node_id, vel, byref(self.pErrorCode))
        self.epos.VCS_HaltVelocityMovement(self.keyhandle, self.cfg.node_id, byref(self.pErrorCode))
