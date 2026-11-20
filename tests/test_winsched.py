import subprocess
import sys

import pytest

from yomibench import winsched


@pytest.mark.skipif(sys.platform != "win32", reason="Windows power throttling")
def test_a_process_opts_out_of_ecoqos():
    code = ("from yomibench import winsched; assert winsched.opt_out_of_ecoqos(); "
            "print(winsched.ecoqos_state())")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "(1, 0)"  # execution speed controlled, never throttled


def test_off_windows_it_does_nothing(monkeypatch):
    monkeypatch.setattr(winsched.sys, "platform", "linux")
    assert winsched.opt_out_of_ecoqos() is False and winsched.ecoqos_state() is None


def test_selfplay_and_gate_processes_opt_out_at_startup(monkeypatch, tmp_path):
    """Both entry points every worker and gate child runs call winsched.opt_out_of_ecoqos first."""
    import importlib.util
    from pathlib import Path

    from yomibench import cli

    calls = []
    monkeypatch.setattr(winsched, "opt_out_of_ecoqos", lambda: calls.append(1) or True)
    cli.main(["selfplay", "--engine", "toy", "--games", "1", "--out", str(tmp_path / "sp"), "--no-ledger"])
    spec = importlib.util.spec_from_file_location("pool_eval", Path(__file__).parents[1] / "tools/pool_eval.py")
    pool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pool)
    pool.main(["--candidate", "random", "--previous-best", "greedy", "--quick", "--pairs", "1", "--characters", "Toy",
               "--seat-preflight", "0", "--results", str(tmp_path / "g.jsonl"), "--no-ledger"])
    assert calls == [1, 1]
    cli.main(["selfplay", "--engine", "toy", "--games", "1", "--out", str(tmp_path / "sp2"), "--no-ledger", "--ecoqos"])
    assert calls == [1, 1]  # --ecoqos keeps Windows' default
