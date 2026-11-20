"""Windows scheduling for throughput processes.

Windows 11 gives windowless background processes EcoQoS (execution-speed power throttling), and on
hybrid CPUs it keeps EcoQoS threads on the efficiency cores. A self-play PC's run (detached,
BelowNormal) ran on its 12 E-cores at 100% while the 8 P-cores sat near 12%; opting the processes
out moved the load onto every core (P-cores 74%, E-cores 82%) and cut the cost of a preview.
`opt_out_of_ecoqos` does that for the calling process (its threads follow); the priority class is
left alone, so the machine stays responsive.
"""
from __future__ import annotations

import sys

PROCESS_POWER_THROTTLING = 4  # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
EXECUTION_SPEED = 0x1  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED


def _kernel32():
    """kernel32 with the handle types declared (GetCurrentProcess's pseudo-handle is -1: as ctypes'
    default int it would be truncated to an invalid handle on 64-bit Python)."""
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    for f in (k32.SetProcessInformation, k32.GetProcessInformation):
        f.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        f.restype = wintypes.BOOL
    return k32


def _state():
    import ctypes

    class State(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]
    return State


def opt_out_of_ecoqos() -> bool:
    """Ask Windows not to throttle this process's execution speed (no EcoQoS, so P-cores too). True
    if it was applied; False on other platforms or if Windows refused."""
    if sys.platform != "win32":
        return False
    import ctypes

    k32 = _kernel32()
    s = _state()(1, EXECUTION_SPEED, 0)  # control execution speed, state off: never throttle
    return bool(k32.SetProcessInformation(k32.GetCurrentProcess(), PROCESS_POWER_THROTTLING, ctypes.byref(s),
                                          ctypes.sizeof(s)))


def ecoqos_state() -> tuple[int, int] | None:
    """(ControlMask, StateMask) of this process's power throttling; None off Windows or on failure."""
    if sys.platform != "win32":
        return None
    import ctypes

    k32 = _kernel32()
    s = _state()(1, 0, 0)
    ok = k32.GetProcessInformation(k32.GetCurrentProcess(), PROCESS_POWER_THROTTLING, ctypes.byref(s), ctypes.sizeof(s))
    return (s.ControlMask, s.StateMask) if ok else None
