from __future__ import annotations

import time

from devices.epos_driver import EposDriver, EposConfig, Direction

LIB = "/home/tomas/thesis/epos/EPOS-Linux-Library-En/EPOS_Linux_Library/lib/intel/x86_64/libEposCmd.so.6.8.1.0"

ep = EposDriver(EposConfig(lib_path=LIB))

try:
    ep.open()
    print("EPOS opened.")
    print("Current (mA):", ep.get_current_mA())

    print("Jog CW for 2s...")
    ep.jog_start(rpm_output=30, direction=Direction.CW)
    time.sleep(2)
    ep.jog_stop()

    time.sleep(0.5)

    print("Jog CCW for 2s...")
    ep.jog_start(rpm_output=30, direction=Direction.CCW)
    time.sleep(2)
    ep.jog_stop()

    print("Done.")

finally:
    # Always stop, even if something crashes
    try:
        ep.jog_stop()
    except Exception:
        pass
    try:
        ep.close()
    except Exception:
        pass
    print("EPOS closed (best effort).")
