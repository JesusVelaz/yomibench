"""Agent-started processes run on the never-shown desktop, their children stay there, and the mod's launcher takes
its client down with it."""
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows desktops")

REPORT_DESKTOP = (
    "import ctypes, sys\n"
    "from ctypes import wintypes\n"
    "u = ctypes.WinDLL('user32'); k = ctypes.WinDLL('kernel32')\n"
    "u.GetThreadDesktop.restype = wintypes.HANDLE\n"
    "u.GetUserObjectInformationW.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,"
    " ctypes.POINTER(wintypes.DWORD)]\n"
    "b = ctypes.create_unicode_buffer(256); n = wintypes.DWORD()\n"
    "u.GetUserObjectInformationW(u.GetThreadDesktop(k.GetCurrentThreadId()), 2, b, 512, ctypes.byref(n))\n"
    "print(b.value, flush=True)\n"
)


def _desktop_of(tmp_path, start):
    out = tmp_path / "out.txt"
    with open(out, "wb") as f:
        p = start([sys.executable, "-c", REPORT_DESKTOP], f)
        p.wait(timeout=60)
    return out.read_text().strip()


def test_popen_hidden_starts_on_the_hidden_desktop(tmp_path):
    from yomibench.game import popen_hidden
    from yomibench.hidden_desktop import DESKTOP

    assert _desktop_of(tmp_path, lambda a, f: popen_hidden(a, stdout=f, creationflags=subprocess.CREATE_NO_WINDOW)) \
        == DESKTOP


def test_children_inherit_the_hidden_desktop(tmp_path):
    from yomibench.hidden_desktop import DESKTOP, popen

    child = f"import subprocess, sys; subprocess.run([sys.executable, '-c', {REPORT_DESKTOP!r}])"
    out = tmp_path / "out.txt"
    with open(out, "wb") as f:
        popen([sys.executable, "-c", child], stdout=f, creationflags=subprocess.CREATE_NO_WINDOW).wait(timeout=60)
    assert out.read_text().strip() == DESKTOP


def test_ending_the_launcher_ends_what_it_started(tmp_path):
    pid_file = tmp_path / "client.pid"
    launcher = subprocess.Popen(
        [sys.executable, "-m", "yomibench.hidden_desktop", "--launch-hidden", "--pid-file", str(pid_file),
         sys.executable, "-c", "import time; time.sleep(120)"], creationflags=subprocess.CREATE_NO_WINDOW)
    deadline = time.monotonic() + 30
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    client = int(pid_file.read_text())

    def alive(pid):
        return str(pid) in subprocess.run(["tasklist", "/fi", f"pid eq {pid}", "/fo", "csv"],
                                          capture_output=True, text=True).stdout

    assert alive(client)
    launcher.kill()
    launcher.wait(timeout=30)
    deadline = time.monotonic() + 10
    while alive(client) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not alive(client)
