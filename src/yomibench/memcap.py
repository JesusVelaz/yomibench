"""A per-process memory cap for self-play children (`selfplay --max-proc-mem-gb G`).

A runaway game (one search whose memory keeps growing) should cost that game, not the machine.
In each `--procs` child a daemon thread samples the process's own memory every `interval`
seconds: private (commit) bytes on Windows, which is what exhausts a machine there (read as
ledger.MemorySampler does, with GetProcessMemoryInfo); resident bytes elsewhere (psutil if
installed, else /proc/self/statm, else getrusage's peak). Past the cap, while a game is in play,
it writes one `dropped-<seed>.json` per game in play into the run's claim folder (the parent reads
them: `collect_drops`), lets the child hand over its ledger counters, and ends the process with
os._exit(EXIT_CODE). The engine's search is native code that a Python exception can't interrupt
mid-call; os._exit needs only this thread to get the GIL, which it does between engine calls (the
batch engine also releases it while previewing). The parent records those seeds as dropped (their
claims stay, so they are never replayed in the run) and starts a replacement child, which goes on
claiming the seeds no child has taken (cli._selfplay_procs)."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

EXIT_CODE = 86  # a child ended by its memory cap (distinct from crashes, 1, and Ctrl-C, 130)
GB = 2 ** 30


def process_memory_bytes() -> int | None:
    """This process's memory now: private (commit) bytes on Windows, resident bytes elsewhere
    (the peak, from getrusage, where nothing better is available). None if unknown."""
    if sys.platform == "win32":
        return _win_private_bytes()
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        pass
    try:
        with open("/proc/self/statm", encoding="ascii") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError, AttributeError):
        pass
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except Exception:  # noqa: BLE001
        return None


def _win_private_bytes() -> int | None:
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS_EX
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                        ("PrivateUsage", ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        get = ctypes.windll.psapi.GetProcessMemoryInfo
        get.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        if not get(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return None
        return int(c.PrivateUsage)
    except Exception:  # noqa: BLE001
        return None


class MemoryCap:
    """The watchdog. `playing` maps each seed in play in this process to its matchup, and `lock`
    guards run_selfplay's bookkeeping (pass both to it: `playing=`, `guard=`). Tripping takes `lock`
    for good, so no game is claimed, recorded or counted while the process ends; `before_exit`
    runs just before it does (the shard's ledger counters)."""

    def __init__(self, cap_gb: float, report_dir: Path, before_exit: Callable[[], None] | None = None,
                 interval: float = 0.5):
        self.cap_gb, self.cap = cap_gb, int(cap_gb * GB)
        self.report_dir, self.before_exit, self.interval = report_dir, before_exit, interval
        self.playing: dict[int, Any] = {}
        self.lock = threading.Lock()

    def start(self) -> "MemoryCap":
        threading.Thread(target=self._run, daemon=True, name="memory-cap").start()
        return self

    def _run(self) -> None:
        while True:
            time.sleep(self.interval)
            try:
                used = process_memory_bytes()
            except Exception:  # noqa: BLE001  (a failed sample is skipped, never fatal)
                continue
            if used is None or used <= self.cap:
                continue
            self.lock.acquire()
            if self.playing:  # between games: nothing to drop, so carry on
                self._trip(used, dict(self.playing))  # never returns
            self.lock.release()

    def _trip(self, used: int, playing: dict[int, Any]) -> None:
        for seed, chars in sorted(playing.items()):
            drop = {"seed": seed, "characters": list(chars), "memory_bytes": used, "cap_gb": self.cap_gb,
                    "pid": os.getpid(), "ppid": os.getppid(), "time": time.time()}
            tmp = self.report_dir / f".dropped-{seed}.{os.getpid()}.tmp"
            try:
                tmp.write_text(json.dumps(drop), encoding="utf-8")
                os.replace(tmp, self.report_dir / f"dropped-{seed}.json")
            except OSError as e:
                print(f"memory cap: could not record seed {seed}: {e}", file=sys.stderr, flush=True)
        if self.before_exit is not None:
            try:
                self.before_exit()
            except Exception:  # noqa: BLE001
                pass
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
        os._exit(EXIT_CODE)


def collect_drops(report_dir: Path) -> list[dict[str, Any]]:
    """The games children dropped at their memory cap since the last call (each file is read once,
    then removed), by seed."""
    out = []
    for f in sorted(report_dir.glob("dropped-*.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
            f.unlink()
        except (OSError, ValueError):
            pass
    return sorted(out, key=lambda d: d["seed"])


def drop_reason(drop: dict[str, Any]) -> str:
    return (f"memory cap {drop['cap_gb']:g} GB exceeded: {drop['memory_bytes'] / GB:.3g} GB, "
            f"matchup {':'.join(drop['characters'])}")
