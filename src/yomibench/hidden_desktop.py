"""Start agent-launched game instances on a separate, never-shown Windows desktop.

No game the harness starts on its own (servers, preview pools, renders, real-game checks, scene exports,
autoplay tests, the multiplayer bot client) should appear on your monitors, not even briefly. This
game build ignores --no-window, so instead of hiding windows after the fact, such processes are created on their own
desktop, "YomiBenchHidden", in the same window station (CreateDesktop + STARTUPINFO.lpDesktop). Nothing on that
desktop is ever shown, and processes started from there inherit it. Your own play stays on the normal desktop.

    proc = popen(args, cwd=..., stdout=file_or_None, creationflags=...)   # Popen-like: pid, poll, wait, kill
    <bot server> --launch-hidden [--pid-file F] EXE ARGS...                # the game mod's bot client (main)

Elsewhere than Windows, `popen` is plain subprocess.Popen.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from ctypes import wintypes

DESKTOP = "YomiBenchHidden"
_desktop_handle = None  # kept open for this process's lifetime, so the desktop exists while we start things

if sys.platform == "win32":
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    u32 = ctypes.WinDLL("user32", use_last_error=True)

    class STARTUPINFOW(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR), ("lpDesktop", wintypes.LPWSTR),
                    ("lpTitle", wintypes.LPWSTR), ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
                    ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD), ("dwXCountChars", wintypes.DWORD),
                    ("dwYCountChars", wintypes.DWORD), ("dwFillAttribute", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD), ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD),
                    ("lpReserved2", ctypes.c_void_p), ("hStdInput", wintypes.HANDLE),
                    ("hStdOutput", wintypes.HANDLE), ("hStdError", wintypes.HANDLE)]

    class PROCESS_INFORMATION(ctypes.Structure):
        _fields_ = [("hProcess", wintypes.HANDLE), ("hThread", wintypes.HANDLE),
                    ("dwProcessId", wintypes.DWORD), ("dwThreadId", wintypes.DWORD)]

    k32.CreateProcessW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p, ctypes.c_void_p, wintypes.BOOL,
                                   wintypes.DWORD, ctypes.c_void_p, wintypes.LPCWSTR, ctypes.POINTER(STARTUPINFOW),
                                   ctypes.POINTER(PROCESS_INFORMATION)]
    k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    u32.CreateDesktopW.restype = wintypes.HANDLE
    u32.CreateDesktopW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.c_void_p, wintypes.DWORD,
                                   wintypes.DWORD, ctypes.c_void_p]
    u32.WaitForInputIdle.argtypes = [wintypes.HANDLE, wintypes.DWORD]


def ensure_desktop() -> str:
    """Create (or open) the hidden desktop; returns the lpDesktop name to start processes on."""
    global _desktop_handle
    if _desktop_handle is None:
        h = u32.CreateDesktopW(DESKTOP, None, None, 0, 0x10000000, None)  # GENERIC_ALL
        if not h:
            raise OSError(ctypes.get_last_error(), "CreateDesktop failed")
        _desktop_handle = h
    return DESKTOP  # on the caller's window station (WinSta0 for an interactive user)


def _cmdline(args) -> str:
    return subprocess.list2cmdline([str(a) for a in args])


def _env_block(env: dict | None):
    if env is None:
        return None
    text = "".join(f"{k}={v}\0" for k, v in env.items()) + "\0"
    return ctypes.create_unicode_buffer(text)


class HiddenProcess:
    """The subset of subprocess.Popen the callers use: pid, returncode, poll, wait, kill, terminate."""

    def __init__(self, pi):
        self._h = pi.hProcess
        self._handle = int(pi.hProcess)  # like Popen's, for yomibench.game's job object
        self.pid = int(pi.dwProcessId)
        self.returncode = None

    def poll(self):
        if self.returncode is None and k32.WaitForSingleObject(self._h, 0) == 0:
            code = wintypes.DWORD()
            k32.GetExitCodeProcess(self._h, ctypes.byref(code))
            self.returncode = int(code.value)
        return self.returncode

    def wait(self, timeout: float | None = None):
        ms = 0xFFFFFFFF if timeout is None else int(timeout * 1000)
        if k32.WaitForSingleObject(self._h, ms) != 0:
            raise subprocess.TimeoutExpired("hidden-desktop process", timeout)
        return self.poll()

    def kill(self):
        if self.poll() is None:
            k32.TerminateProcess(self._h, 1)

    terminate = kill

    def __del__(self):
        try:
            k32.CloseHandle(self._h)
        except Exception:  # noqa: BLE001
            pass


def popen(args, cwd=None, env=None, stdout=None, stderr=None, creationflags: int = 0, **_ignored):
    """Start `args` on the hidden desktop. `stdout`/`stderr`: None or subprocess.DEVNULL (discarded), a file object,
    or subprocess.STDOUT for stderr (same as stdout). Other Popen keywords are ignored."""
    if sys.platform != "win32":
        return subprocess.Popen(args, cwd=cwd, env=env, stdout=stdout, stderr=stderr)
    import msvcrt

    desk = ensure_desktop()
    si = STARTUPINFOW()
    si.cb = ctypes.sizeof(STARTUPINFOW)
    si.lpDesktop = desk
    keep = []

    def handle_for(f):
        if f is None or f == subprocess.DEVNULL:
            f = open(os.devnull, "wb")
            keep.append(f)
        h = msvcrt.get_osfhandle(f.fileno())
        os.set_handle_inheritable(h, True)
        return h

    out = handle_for(stdout)
    err = out if stderr == subprocess.STDOUT else handle_for(stderr)
    si.dwFlags = 0x100  # STARTF_USESTDHANDLES
    si.hStdOutput, si.hStdError, si.hStdInput = out, err, handle_for(None)
    pi = PROCESS_INFORMATION()
    flags = creationflags | 0x00000400  # CREATE_UNICODE_ENVIRONMENT
    block = _env_block(env)
    ok = k32.CreateProcessW(None, ctypes.create_unicode_buffer(_cmdline(args)), None, None, True, flags,
                            block, None if cwd is None else str(cwd), ctypes.byref(si), ctypes.byref(pi))
    for f in keep:
        f.close()
    if not ok:
        raise OSError(ctypes.get_last_error(), f"CreateProcess on the hidden desktop failed: {args[0]}")
    k32.CloseHandle(pi.hThread)
    return HiddenProcess(pi)


def _kill_on_close_job(proc):
    """A job object that ends `proc` when this process ends (its handle closes), however that happens."""
    class LIMITS(ctypes.Structure):  # JOBOBJECT_EXTENDED_LIMIT_INFORMATION (64-bit layout via native types)
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD), ("io", ctypes.c_uint64 * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    job = k32.CreateJobObjectW(None, None)
    info = LIMITS()
    info.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))  # ExtendedLimitInformation
    k32.AssignProcessToJobObject(job, int(getattr(proc, "_h", None) or proc._handle))
    return job


def main(argv=None) -> int:
    """The game mod's way to start its multiplayer bot client on the hidden desktop (GDScript can't name a desktop):
    `--launch-hidden [--pid-file F] EXE ARGS...` starts EXE there at below-normal priority, writes its pid to F,
    and stays until it exits. Ending this launcher ends EXE too (a kill-on-close job), so the mod stops the client
    by stopping the launcher, as before."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--launch-hidden"]:
        argv = argv[1:]
    pid_file = None
    if argv[:1] == ["--pid-file"] and len(argv) >= 2:
        pid_file, argv = argv[1], argv[2:]
    if not argv:
        print("usage: --launch-hidden [--pid-file F] EXE [ARGS...]", file=sys.stderr)
        return 2
    from yomibench.game import popen_hidden  # this desktop, else its window-hiding fallback

    p = popen_hidden(argv, cwd=os.path.dirname(argv[0]) or None,
                     creationflags=0x00004000 | 0x08000000)  # BELOW_NORMAL, CREATE_NO_WINDOW
    if sys.platform == "win32":
        _job = _kill_on_close_job(p)  # noqa: F841  (held for this process's lifetime)
    if pid_file:
        with open(pid_file + ".tmp", "w", encoding="ascii") as f:
            f.write(str(p.pid))
        os.replace(pid_file + ".tmp", pid_file)  # never seen half-written
    print(p.pid, flush=True)
    while p.poll() is None:
        time.sleep(0.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
