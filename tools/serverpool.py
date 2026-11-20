"""A pool of real-game servers for the test tools: start, restart after a crash, sample memory.

    from serverpool import Pool
    with Pool(8, base_port=7900) as pool:
        engine = pool.engine(0)          # yomibench.engine.Engine on the first server
        ...
        pool.restart(0)                  # after a crash
        pool.memory_mb()                 # working set per process

Each server gets its own run dir (~/.yomibench/pool/PORT) and user data dir
(%APPDATA%/YomiBenchPoolPORT), so its engine log (`log_path(i)`, written as it happens) belongs to it
alone: tools can tell which requests a script error followed. Stdout is discarded (the same text
goes to that log). Uses yomibench.game for building run dirs.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

from yomibench import game
from yomibench.engine import Engine, Tcp


def _pmc(pid: int) -> tuple[float, float] | None:
    """(working set MB, peak working set MB), Windows only."""
    if sys.platform != "win32":
        return None
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    h = ctypes.windll.kernel32.OpenProcess(0x0410, False, pid)
    if not h:
        return None
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    try:
        if not ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return None
        return pmc.WorkingSetSize / 2**20, pmc.PeakWorkingSetSize / 2**20
    finally:
        ctypes.windll.kernel32.CloseHandle(h)


def _cpu_seconds(pid: int) -> float | None:
    """User + kernel CPU seconds of a process (Windows only)."""
    if sys.platform != "win32":
        return None
    from ctypes import wintypes

    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return None
    try:
        c, e, k, u = (wintypes.FILETIME() for _ in range(4))
        if not ctypes.windll.kernel32.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(k), ctypes.byref(u)):
            return None
        ft = lambda f: (f.dwHighDateTime << 32 | f.dwLowDateTime) / 1e7  # noqa: E731
        return ft(k) + ft(u)
    finally:
        ctypes.windll.kernel32.CloseHandle(h)


class Pool:
    def __init__(self, count: int, base_port: int = 7900, game_dir: str | None = None,
                 extra_args: list[str] | None = None, run_dir: Path | None = None):
        # run_dir: use an already built run dir as is (e.g. one holding other script versions);
        # all servers then share it and its user data dir.
        self.ports = [base_port + i for i in range(count)]
        if run_dir is not None:
            self.run_dirs = [Path(run_dir)] * count
            self.user_dirs = ["YomiBenchServer"] * count
        else:
            gd = game.find_game_dir(game_dir)
            self.user_dirs = [f"YomiBenchPool{p}" for p in self.ports]
            self.run_dirs = [game.build_run_dir(gd, Path.home() / ".yomibench" / "pool" / str(p), u)
                             for p, u in zip(self.ports, self.user_dirs)]
        self.extra_args = extra_args or []
        self.procs: list = [None] * count
        self.restarts = [0] * count
        self.startup_failures: list[dict] = []
        try:
            for i in range(count):
                self._start(i)
            for i in range(count):
                self._wait(i)
        except BaseException:
            self.close()
            raise

    def _wait(self, i: int, tries: int = 3) -> None:
        # Startups occasionally die (seen: access violation while many servers start at once);
        # retry, and keep a record.
        for attempt in range(tries):
            try:
                game.wait_ready(self.ports[i], self.procs[i])
                return
            except RuntimeError as e:
                self.startup_failures.append({"port": self.ports[i], "error": str(e), "time": time.time()})
                if attempt == tries - 1:
                    raise
                time.sleep(1.0)
                self._start(i)

    def _start(self, i: int) -> None:
        port = self.ports[i]
        if game.port_in_use(port):
            raise RuntimeError(f"port {port} is already in use")
        rd = self.run_dirs[i]
        args = [str(rd / game.EXE), "--no-window", "--disable-render-loop", "--audio-driver", "Dummy",
                f"--yomi-port={port}", *self.extra_args]
        # Stderr carries native panic messages (the engine log only says "check stderr").
        with open(self.err_path(i), "ab") as err:
            self.procs[i] = game.popen_hidden(args, cwd=rd, stdout=subprocess.DEVNULL, stderr=err)

    def alive(self, i: int) -> bool:
        return self.procs[i].poll() is None

    def restart(self, i: int) -> None:
        p = self.procs[i]
        if p.poll() is None:
            p.kill()
        p.wait()
        self.restarts[i] += 1
        deadline = time.monotonic() + 10
        while game.port_in_use(self.ports[i]) and time.monotonic() < deadline:
            time.sleep(0.2)
        self._start(i)
        self._wait(i)

    def engine(self, i: int, timeout: float = 600.0) -> Engine:
        return Engine(Tcp("127.0.0.1", self.ports[i], timeout=timeout))

    def err_path(self, i: int) -> Path:
        return self.run_dirs[i] / "stderr.log"

    def log_path(self, i: int) -> Path:
        """The engine log of server i's current process (Godot rotates it at each start)."""
        return Path(os.environ.get("APPDATA", "")) / self.user_dirs[i] / "logs" / "godot.log"

    def exit_code(self, i: int) -> int | None:
        return self.procs[i].poll()

    def memory_mb(self) -> list[tuple[float, float] | None]:
        return [_pmc(p.pid) if p.poll() is None else None for p in self.procs]

    def cpu_seconds(self) -> list[float | None]:
        return [_cpu_seconds(p.pid) if p.poll() is None else None for p in self.procs]

    def close(self) -> None:
        for p in self.procs:
            if p is not None and p.poll() is None:
                p.kill()
        for p in self.procs:
            if p is not None:
                p.wait()

    def __enter__(self) -> "Pool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()



class LogTail:
    """New lines of a growing log file since the last call (restarts at 0 if the file shrinks)."""

    def __init__(self, path: Path):
        self.path = path
        self.pos = path.stat().st_size if path.exists() else 0

    def reset(self) -> None:
        self.pos = 0

    def read(self) -> list[str]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self.pos:
            self.pos = 0
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            data = f.read()
        cut = data.rfind(b"\n") + 1   # whole lines only
        self.pos += cut
        return data[:cut].decode("utf-8", "replace").splitlines()
